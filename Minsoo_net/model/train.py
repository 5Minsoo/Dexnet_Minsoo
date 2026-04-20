# -*- coding: utf-8 -*-
"""
Dex-Net 2.0 학습 스크립트 (PyTorch).

zarr 데이터셋 구조:
    root/
      object1/
        pose1/
          image: (H, W) or (H, W, 1)  — depth image
          depth: scalar                — gripper depth
          label: scalar                — grasp quality (연속값)
        pose2/ ...
      object2/ ...

사용법:
    python train.py --data /path/to/data.zarr --output ./output
"""
import argparse
import logging
import os
import time
from datetime import datetime
import cv2
import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import zarr
import yaml
from model import DexNet2
from focal_loss import FocalLoss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
logging.basicConfig(
    format="[%(name)s %(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("train")
import json
with open('/home/minsoo/Dexnet_Minsoo/Minsoo_net/config/master_config.yaml') as f:
    config=yaml.safe_load(f)
# ══════════════════════════════════════════════════════════════════════
#  하드코딩된 학습 설정 (원본 YAML 기준)
# ══════════════════════════════════════════════════════════════════════
CFG = dict(
    # ── 학습 ──
    train_batch_size=64,
    val_batch_size=64,
    num_epochs=25,
    train_pct=0.8,

    # ── optimizer ──
    base_lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,     # L2 regularization 
    decay_rate=0.95,
    decay_step_multiplier=1.5,
    drop_rate=0.2,

    # ── 라벨 ──
    metric_thresh=config.get("quality_threshold",0.002) ,       # label > thresh → positive (1)

    # ── 로깅 / 저장 ──
    eval_frequency=1,           # epoch 단위
    save_frequency=1,           # epoch 단위
    log_frequency=10000,           # step 단위


    # ── 전처리 ──
    num_random_files=10000,     # mean/std 계산 시 사용할 최대 샘플 수

    # ── 기타 ──
    seed=24098,
    alpha=[0.25,0.75],
    loss="CE"
    )

def init_weights(m):
    # Conv2d 또는 Linear 레이어인 경우에만 적용
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # He 정규분포 초기화
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # 편향(bias)이 존재하면 0으로 초기화
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
# ══════════════════════════════════════════════════════════════════════
#  Zarr Dataset
# ══════════════════════════════════════════════════════════════════════
class DexNetZarrDataset(Dataset):
    """
    zarr 파일에서 (image, depth, label) 트리플을 평탄화하여 로드.

    expected zarr structure:
        object_name / pose_name / {image, depth, label}
    """

    def __init__(self, zarr_path, indices=None, im_height=32, im_width=32,
                 metric_thresh=0.002, cfg=None,
                 im_mean=0.0, im_std=1.0, pose_mean=0.0, pose_std=1.0):
        super().__init__()
        self.root = zarr.open(str(zarr_path), mode="r")
        self.im_height = im_height
        self.im_width = im_width
        self.metric_thresh = metric_thresh ##어차피 특정 확률 필터링은 데이터 생성때 함. 이진화는 그냥 확률이 0이 아니기만 하면 다 1로 처리.
        self.cfg = cfg or CFG

        self.im_mean = im_mean
        self.im_std = im_std
        self.pose_mean = pose_mean
        self.pose_std = pose_std

        # 모든 (object, pose, grasp) 경로를 평탄화
        self.paths = []
        self.success = 0  
        self.total_samples = 0 # 전체 샘플 수를 저장할 변수 추가

        # ── 물체별 정답 비율 역수 가중치 ──
        obj_success_rates = {}
        for obj_key in self.root.keys():
            obj_group = self.root[obj_key]
            obj_total = 0
            obj_success = 0
            for pose_key in obj_group.keys():
                labels = np.array(obj_group[pose_key]["labels"])
                obj_total += len(labels)
                obj_success += np.sum(labels > metric_thresh)
                num_grasps=len(labels)
                for grasp_idx in range(num_grasps):
                    self.paths.append((obj_key, pose_key, grasp_idx))

            rate = obj_success / obj_total if obj_total > 0 else 1e-6
            obj_success_rates[obj_key] = max(rate, 1e-6)
            self.total_samples+=obj_total
            self.success+=obj_success

        # 5. 최종 결과 출력 (전체 샘플 수에서 성공을 뺌)
        print("-" * 30)
        print(f"데이터셋 통계 ({datetime.now().strftime('%m/%d %H:%M')})")
        print(f"전체 Data : {self.total_samples}개")
        print(f"성공 Data : {self.success}개 ({(self.success/self.total_samples)*100:.1f}%)")
        print(f"실패 Data : {self.total_samples - self.success}개")
        print("-" * 30)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 이제 idx를 통해 obj, pose, 그리고 N개 중 몇 번째인지(grasp_idx)까지 알 수 있습니다!
        obj_key, pose_key, grasp_idx = self.paths[idx]
        group = self.root[obj_key][pose_key]

        # ── 이미지 로드 ──
        # 통째로 다 불러오지 말고, zarr에서 필요한 그 1장만 쏙 빼옵니다. (속도 훨씬 빠름)
        image = np.array(group["images"][grasp_idx], dtype=np.float32)
        # 리사이즈 (차원이 [H, W]가 되었으므로 shape[0], shape[1]로 검사)
        if image.shape[0] != self.im_height or image.shape[1] != self.im_width:
            image = cv2.resize(image, (self.im_width, self.im_height), interpolation=cv2.INTER_CUBIC)

        # ── depth / label ──
        depth = float(np.array(group["gripper_depth"][grasp_idx]))
        label = float(np.array(group["labels"][grasp_idx]))

        # ── 정규화 ──
        image = (image - self.im_mean) / (self.im_std + 1e-10)
        depth = (depth - self.pose_mean) / (self.pose_std + 1e-10)

        # ── 라벨 이진화 ──
        binary_label = 1 if label > self.metric_thresh else 0

        # (1, H, W) tensor
        image_t = torch.from_numpy(image[np.newaxis, :, :]).float()
        depth_t = torch.tensor([depth], dtype=torch.float32)
        label_t = torch.tensor(binary_label, dtype=torch.long)

        return image_t, depth_t, label_t


# ══════════════════════════════════════════════════════════════════════
#  전처리: mean / std 계산
# ══════════════════════════════════════════════════════════════════════
def compute_data_stats(zarr_path, train_indices, im_height, im_width,
                       max_samples=10000):
    """학습 데이터에서 image mean/std, pose mean/std를 계산."""
    log.info("Computing dataset statistics...")
    root = zarr.open(str(zarr_path), mode="r")

    # 평탄화된 경로
    all_paths = []
    for obj_key in root.keys():
        for pose_key in root[obj_key].keys():
            num_grasps=root[obj_key][pose_key]["labels"].shape[0]
            for grasps in range(num_grasps):
                all_paths.append((obj_key,pose_key,grasps))

    train_paths = [all_paths[i] for i in train_indices]

    # 샘플링
    n = min(len(train_paths), max_samples)
    sampled = np.random.choice(len(train_paths), size=n, replace=False)

    # ── 이미지 mean/std ──
    im_sum = 0.0
    im_sq_sum = 0.0
    num_pixels = 0

    pose_values = []

    for k, idx in enumerate(sampled):
        if k % 2000 == 0:
            log.info(f"  Processing {k}/{n} samples...")
        obj_key, pose_key, grasp_num = train_paths[idx]
        group = root[obj_key][pose_key]

        image = np.array(group["images"][grasp_num], dtype=np.float64)
        if image.ndim == 3:
            image = image[:, :, 0]

        if image.shape[0] != im_height or image.shape[1] != im_width:
            print('저장된 Image 사이즈가 모델 입력과 다릅니다')
            image = cv2.resize(image.astype(np.float32),
                               (im_width, im_height),
                               interpolation=cv2.INTER_CUBIC).astype(np.float64)

        im_sum += image.sum()
        im_sq_sum += (image ** 2).sum()
        num_pixels += image.size

        depth = float(np.array(group["gripper_depth"][grasp_num]))
        pose_values.append(depth)

    im_mean = im_sum / num_pixels
    im_std = np.sqrt(im_sq_sum / num_pixels - im_mean ** 2)
    if im_std < 1e-10:
        im_std = 1.0

    pose_arr = np.array(pose_values, dtype=np.float64)
    pose_mean = pose_arr.mean()
    pose_std = pose_arr.std()
    if pose_std < 1e-10:
        pose_std = 1.0

    log.info(f"  im_mean={im_mean:.6f}, im_std={im_std:.6f}")
    log.info(f"  pose_mean={pose_mean:.6f}, pose_std={pose_std:.6f}")

    return (float(im_mean), float(im_std),
            float(pose_mean), float(pose_std))


# ══════════════════════════════════════════════════════════════════════
#  Validation
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device):
    """Validation error rate + loss."""
    model.eval()
    criterion=nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, poses, labels in loader:
        images = images.to(device)
        poses = poses.to(device)
        labels = labels.to(device)

        logits = model(images, poses)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    error_rate = 1.0 - correct / max(total, 1)
    return avg_loss, error_rate


# ══════════════════════════════════════════════════════════════════════
#  학습 루프
# ══════════════════════════════════════════════════════════════════════
def train(args):
    cfg = CFG.copy()

    # CLI 오버라이드
    if args.epochs is not None:
        cfg["num_epochs"] = args.epochs
    if args.lr is not None:
        cfg["base_lr"] = args.lr
    if args.batch_size is not None:
        cfg["train_batch_size"] = args.batch_size
        cfg["val_batch_size"] = args.batch_size
    cfg["metric_thresh"]=args.thresh
    cfg["alpha"]=args.alpha
    cfg["loss"]=args.loss
    # 시드 설정
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── zarr 열어서 전체 샘플 수 파악 ──
    root = zarr.open(str(args.data), mode="r")
    all_paths =0
    for obj_key in root.keys():
        for pose_key in root[obj_key].keys():
            all_paths+=root[obj_key][pose_key]["labels"].shape[0]

    num_total = all_paths
    log.info(f"Total samples (전체 Pose 개수): {num_total}")

    # ── Train / Val split ──
    indices = np.arange(num_total)
    np.random.shuffle(indices)
    split = int(num_total * cfg["train_pct"])
    train_indices = indices[:split].tolist()
    val_indices = indices[split:].tolist()
    log.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # ── 통계 계산 ──
    im_mean, im_std, pose_mean, pose_std = compute_data_stats(
        args.data, train_indices, args.im_height, args.im_width,
        max_samples=cfg["num_random_files"])

    # ── 데이터셋 ──

    full_ds = DexNetZarrDataset(
    args.data,
    im_height=args.im_height, im_width=args.im_width,
    metric_thresh=cfg["metric_thresh"], cfg=cfg,
    im_mean=im_mean, im_std=im_std,
    pose_mean=pose_mean, pose_std=pose_std)

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train_batch_size"], # shuffle 대신 sampler 사용
        num_workers=12, pin_memory=True, drop_last=True, shuffle=True)
    
    steps_per_epoch = len(train_loader)   # = len(train_ds) // batch_size (drop_last=True)

    cfg["decay_step"] = int(cfg["decay_step_multiplier"] * steps_per_epoch)

    log.info(f"Steps per epoch: {steps_per_epoch}")
    log.info(f"Decay step: {cfg['decay_step']} "
            f"(= {cfg['decay_step_multiplier']} × {steps_per_epoch})")
    
    val_loader = DataLoader(
        val_ds, batch_size=cfg["val_batch_size"],
        shuffle=False, num_workers=12, pin_memory=True)

    # ── 모델 ──
    model = DexNet2(im_height=args.im_height, im_width=args.im_width)
    model.im_mean = im_mean
    model.im_std = im_std
    model.pose_mean = np.array([pose_mean], dtype=np.float32)
    model.pose_std = np.array([pose_std], dtype=np.float32)
    model.apply(init_weights)
    model.to(device)

    # ── Loss / Optimizer / Scheduler ──    
    if cfg["loss"]=="FL":
        criterion = FocalLoss(task_type="multi-class",num_classes=2,alpha=cfg["alpha"])
    elif cfg['loss']=="CE":
        # num_pos = train_ds.success
        # num_neg = train_ds.total_samples - num_pos
        # weight = torch.tensor([1.0, num_neg / num_pos], dtype=torch.float32).to(device)
        # criterion = nn.CrossEntropyLoss(weight=weight)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"CE or FL 넣어야 함. error: {cfg['loss']}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["base_lr"],
        weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg["decay_rate"])

    # ── 출력 디렉토리 ──
    os.makedirs(args.output, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output, "train.log"))
    file_handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
    log.addHandler(file_handler)
    # ── 학습 ──
    log.info("=" * 60)
    log.info("Starting training")
    log.info("=" * 60)

    global_step = 0
    best_val_error = 1.0
    loss_graph=[]
    train_loss_graph=[]
    val_loss_graph=[]

    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
    log.info(f"Config: {cfg}")

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for batch_idx, (images, poses, labels) in enumerate(train_loader):
            images = images.to(device)
            poses = poses.to(device)
            labels = labels.to(device)

            logits = model(images, poses, drop_rate=cfg["drop_rate"])
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # 통계
            preds = logits.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)

            epoch_loss += loss.item() * batch_total
            epoch_correct += batch_correct
            epoch_total += batch_total
            global_step += 1

            loss_graph.append(loss.item())
            if global_step % cfg["log_frequency"] == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info(
                    f"  [epoch {epoch} step {global_step}] "
                    f"loss={loss.item():.4f} "
                    f"batch_acc={batch_correct/batch_total:.3f}  "
                    f"lr={lr_now:.6f}")
                
            if global_step % cfg["decay_step"] == 0:
                scheduler.step()
                log.info(f"  >> LR decayed at step {global_step}: "
                        f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # ── Epoch 요약 ──
        train_loss = epoch_loss / max(epoch_total, 1)
        train_loss_graph.append(train_loss)
        train_error = 1.0 - epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch}/{cfg['num_epochs']}  "
            f"train_loss={train_loss:.4f}  train_error={train_error:.4f}  "
            f"time={elapsed:.1f}s")

        # ── Validation ──
        if epoch % cfg["eval_frequency"] == 0 or epoch == cfg["num_epochs"]:
            val_loss, val_error = evaluate(model, val_loader, device)
            val_loss_graph.append(val_loss)
            log.info(
                f"  >> val_loss={val_loss:.4f}  val_error={val_error:.4f}")
            
            if val_error < best_val_error:
                best_val_error = val_error
                model.save(os.path.join(args.output, "best.pt"))
                log.info(f"  >> New best model saved (val_error={val_error:.4f})")

        # ── 체크포인트 저장 ──
        if epoch % cfg["save_frequency"] == 0:
            ckpt_path = os.path.join(args.output, f"epoch_{epoch:03d}.pt")
            model.save(ckpt_path)
            log.info(f"  Checkpoint saved: {ckpt_path}")
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.plot(train_loss_graph, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Train_Loss')
        plt.legend()
        if val_loss_graph:
            plt.subplot(1,3,2)
            plt.plot(val_loss_graph, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Validation_Loss')
            plt.legend()
        plt.subplot(1,3,3)
        plt.plot(loss_graph,label="loss graph")
        plt.xlabel('Batch')
        plt.ylabel('Batch_Loss')
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'training_curve.png'))
        plt.close() 

    # ── 최종 저장 ──
    model.save(os.path.join(args.output, "final.pt"))
    log.info("=" * 60)
    log.info(f"Training complete. Best val_error={best_val_error:.4f}")
    log.info(f"Model saved to {args.output}/")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    now = datetime.now().strftime("%m-%d_%H-%M")
    parser = argparse.ArgumentParser(description="Dex-Net 2.0 GQ-CNN 학습")
    parser.add_argument("--data", type=str, default=config.get('zarr_path'))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--im-height", type=int, default=32)
    parser.add_argument("--im-width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--thresh", type=float, default=0.002)
    parser.add_argument("--alpha", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--loss", type=str, default="CE")
    args = parser.parse_args()

    if args.output is None:
        data_name = os.path.splitext(os.path.basename(args.data))[0]
        alpha_str = f"{args.alpha[0]}_{args.alpha[1]}"
        args.output = f"./output/{now}_{data_name}_{args.loss}_th{args.thresh}_a{alpha_str}"

    train(args)


if __name__ == "__main__":
    main()