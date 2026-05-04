from collections import defaultdict
import json
import random

import numpy as np
import zarr
import torch
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve, auc, average_precision_score)    
from tqdm import tqdm               

from Minsoo_net.model.model import DexNet2
from Minsoo_net.model.train import DexNetZarrDataset

seed=24098
zarr_path='/home/minsoo/Dexnet_Minsoo/grasp_dataset_ABC.zarr'
model_path='/home/minsoo/Dexnet_Minsoo/output/Dexnet_original/model.pt'
metric_thresh=0.002
train_split=0.8
thresh = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=DexNet2.load(model_path)
model.to(device)

np.random.seed(seed)
torch.manual_seed(seed)

root=zarr.open(zarr_path)
all_paths = []

success = 0  
total_samples = 0

obj_keys = list(root.keys())
total_poses = sum(len(list(root[k].keys())) for k in obj_keys)
with tqdm(total=total_poses, desc="Scanning poses") as pbar:
    for obj_key in root.keys():
        obj_group = root[obj_key]
        for pose_key in obj_group.keys():
            num_grasps = np.array(obj_group[pose_key]["labels"]).shape[0]
            total_samples += num_grasps
            all_paths.extend((obj_key, pose_key, i) for i in range(num_grasps))
            pbar.update(1)

def visualize_first_conv_filters(model, save_path="conv1_filters.png"):
    """첫 번째 conv 층의 필터(weight) 시각화"""
    first_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break
    
    weights = first_conv.weight.data.cpu().numpy()
    n_filters, n_channels, kh, kw = weights.shape
    print(f"필터 형태: {weights.shape}")
    
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    
    for i in range(n_filters):
        ax = axes.flat[i] if n_rows > 1 else axes[i]
        if n_channels == 1:
            f = weights[i, 0]
        else:
            f = weights[i].mean(axis=0)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f, cmap='gray')
        ax.set_title(f"#{i}", fontsize=8)
        ax.axis('off')
    
    for i in range(n_filters, n_rows * n_cols):
        ax = axes.flat[i] if n_rows > 1 else axes[i]
        ax.axis('off')
    
    fig.suptitle(f"First Conv Filters ({n_filters} filters, {kh}x{kw})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"저장됨: {save_path}")

def visualize_first_conv_features(model, image, pose, device, save_path="conv1_features.png"):
    """첫 번째 conv 층의 출력(feature map) 시각화"""
    first_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break
    
    feature_maps = []
    def hook(module, input, output):
        feature_maps.append(output.detach().cpu())
    
    handle = first_conv.register_forward_hook(hook)
    
    model.eval()
    with torch.no_grad():
        img_input = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
        pose_input = pose.unsqueeze(0).to(device) if pose.dim() == 1 else pose.to(device)
        _ = model.predict(img_input, pose_input)
    
    handle.remove()
    
    fmaps = feature_maps[0].squeeze(0).numpy()
    n_filters = fmaps.shape[0]
    
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    
    axes[0, 0].imshow(image.squeeze().cpu(), cmap='gray')
    axes[0, 0].set_title("Input")
    axes[0, 0].axis('off')
    for j in range(1, n_cols):
        axes[0, j].axis('off')
    
    for i in range(n_filters):
        row = (i // n_cols) + 1
        col = i % n_cols
        ax = axes[row, col]
        fmap = fmaps[i]
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        ax.imshow(fmap, cmap='viridis')
        ax.set_title(f"#{i}", fontsize=8)
        ax.axis('off')
    
    for i in range(n_filters + n_cols, n_rows * n_cols):
        axes.flat[i].axis('off')
    
    fig.suptitle(f"First Conv Feature Maps", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"저장됨: {save_path}")

dataset=DexNetZarrDataset(zarr_path=zarr_path, metric_thresh=metric_thresh,paths=all_paths)

indices=np.arange(total_samples)
np.random.shuffle(indices)  
split = int(total_samples * train_split)
train_indices = indices[:split].tolist()
val_indices = indices[split:].tolist()

train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)
val_loader = DataLoader(
    val_ds, batch_size=64,
    shuffle=False, num_workers=8, pin_memory=True)

tp_list, fp_list, tn_list, fn_list = [], [], [], []
results = defaultdict(list)
model.eval()

with torch.no_grad():
    for idx, (images,poses,labels) in enumerate(val_loader):
        images = images.to(device)
        poses = poses.to(device)
        labels = labels.to(device)
        probs=model.predict_success(images,poses)
        pos_probs = torch.from_numpy(probs).to(device)
        pred_pos = pos_probs > thresh
        label_pos = labels == 1.0
        
        masks = {
            'TP': pred_pos & label_pos,
            'FP': pred_pos & ~label_pos,
            'TN': ~pred_pos & ~label_pos,
            'FN': ~pred_pos & label_pos,
        }
        
        for category, mask in masks.items():
            for i in mask.nonzero(as_tuple=True)[0]:
                results[category].append({
                    'prob': probs[i].item(),
                    'label': labels[i].item(),
                    'image': images[i].cpu(),
                    'pose': poses[i].cpu(),
                })

# 통계 출력
total = sum(len(v) for v in results.values())
for cat in ['TP', 'FP', 'TN', 'FN']:
    n = len(results[cat])
    print(f"{cat}: {n} ({n/total*100:.1f}%)")

tp, fp, tn, fn = [len(results[c]) for c in ['TP', 'FP', 'TN', 'FN']]
accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0 
print(f"\nAccuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")                                                           

all_probs = []
all_labels = []
for cat in ['TP', 'FP', 'TN', 'FN']:
    for s in results[cat]:
        all_probs.append(s['prob'])
        all_labels.append(s['label'])
all_probs = np.array(all_probs)
all_labels = np.array(all_labels).astype(int)

# PR Curve
prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
ap_score = average_precision_score(all_labels, all_probs)

plt.figure(figsize=(8, 6))
plt.plot(rec_curve, prec_curve, linewidth=2, color='b', label=f"AP={ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.savefig("precision_recall.png", dpi=100)
plt.close()
print(f"AP score: {ap_score:.3f}")

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc_score = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, color='g', label=f"AUC={auc_score:.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.savefig("roc.png", dpi=100)
plt.close()
print(f"AUC score: {auc_score:.3f}")

# ─────────────────────────────────────────────
# ★ 추가: 요약 통계 JSON 저장
# ─────────────────────────────────────────────
summary_stats = {
    "model_path": model_path,
    "thresh": thresh,
    "total_samples": int(total),
    "TP": int(tp),
    "FP": int(fp),
    "TN": int(tn),
    "FN": int(fn),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "ap_score": float(ap_score),
    "auc_score": float(auc_score),
    "error_rate": float(1 - accuracy),
}

with open("val_stats.json", "w") as f:
    json.dump(summary_stats, f, indent=4, sort_keys=True)
print(f"통계 저장됨: val_stats.json")

# ─────────────────────────────────────────────
# 카테고리별 시각화 (기존 코드 그대로)
# ─────────────────────────────────────────────
for category in ['TP', 'FP', 'TN', 'FN']:
    samples = results[category]
    if len(samples) == 0:
        continue

    s = samples[0]
    visualize_first_conv_features(
        model, s['image'], s['pose'], device,
        save_path=f"conv1_features_{category}.png"
    )

    n_show = min(16, len(samples))
    chosen = random.sample(samples, n_show)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for ax, s in zip(axes.flat, chosen):
        ax.imshow(s['image'].squeeze(), cmap='gray')
        ax.set_title(f"P:{s['prob']:.2f} L:{s['label']:.2f}")
        ax.axis('off')

    fig.suptitle(f"{category} samples (total: {len(samples)})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{category}_samples.png", dpi=100)
    plt.close(fig)

visualize_first_conv_filters(model)