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

import cv2
import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import zarr

from model import DexNet2

logging.basicConfig(
    format="[%(name)s %(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("train")


# ══════════════════════════════════════════════════════════════════════
#  하드코딩된 학습 설정 (원본 YAML 기준)
# ══════════════════════════════════════════════════════════════════════
CFG = dict(
    # ── 학습 ──
    train_batch_size=64,
    val_batch_size=64,
    num_epochs=50,
    train_pct=0.8,

    # ── optimizer ──
    base_lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,        # L2 regularization
    decay_rate=0.95,
    decay_step_multiplier=0.66, # epoch 단위
    drop_rate=0.0,

    # ── 라벨 ──
    metric_thresh=0.002,        # label > thresh → positive (1)

    # ── 로깅 / 저장 ──
    eval_frequency=5,           # epoch 단위
    save_frequency=5,           # epoch 단위
    log_frequency=50,           # step 단위

    # ── 데이터 증강 ──
    multiplicative_denoising=True,
    gamma_shape=1000.0,

    gaussian_process_denoising=True,
    gaussian_process_rate=0.5,
    gaussian_process_sigma=0.005,
    gaussian_process_scaling_factor=4.0,

    symmetrize=True,

    # ── 전처리 ──
    num_random_files=10000,     # mean/std 계산 시 사용할 최대 샘플 수

    # ── 기타 ──
    seed=24098,
)


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
                 metric_thresh=0.002, augment=False, cfg=None,
                 im_mean=0.0, im_std=1.0, pose_mean=0.0, pose_std=1.0):
        super().__init__()
        self.root = zarr.open(str(zarr_path), mode="r")
        self.im_height = im_height
        self.im_width = im_width
        self.metric_thresh = metric_thresh
        self.augment = augment
        self.cfg = cfg or CFG

        self.im_mean = im_mean
        self.im_std = im_std
        self.pose_mean = pose_mean
        self.pose_std = pose_std

        # 모든 (object, pose) 경로를 평탄화
        self.paths = []  # list of (obj_key, pose_key)
        for obj_key in self.root.keys():
            obj_group = self.root[obj_key]
            for pose_key in obj_group.keys():
                self.paths.append((obj_key, pose_key))

        # 인덱스 필터링 (train/val split)
        if indices is not None:
            self.paths = [self.paths[i] for i in indices]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        obj_key, pose_key = self.paths[idx]
        group = self.root[obj_key][pose_key]

        # ── 이미지 로드 ──
        image = np.array(group["image"], dtype=np.float32)
        if image.ndim == 3:
            image = image[:, :, 0]  # (H, W, 1) → (H, W)

        # 리사이즈 (필요한 경우)
        if image.shape[0] != self.im_height or image.shape[1] != self.im_width:
            image = cv2.resize(image, (self.im_width, self.im_height),
                               interpolation=cv2.INTER_CUBIC)

        # ── depth / label ──
        depth = float(np.array(group["depth"]))
        label = float(np.array(group["label"]))

        # ── 증강 ──
        if self.augment:
            image = self._augment(image)

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

    # ── 데이터 증강 ──────────────────────────────────────────────────
    def _augment(self, image):
        """원본 TF 코드와 동일한 증강 파이프라인."""
        h, w = image.shape

        # 1) Multiplicative denoising (gamma noise)
        if self.cfg["multiplicative_denoising"]:
            gamma_shape = self.cfg["gamma_shape"]
            mult = ss.gamma.rvs(gamma_shape, scale=1.0 / gamma_shape)
            image = image * mult

        # 2) Gaussian process denoising (correlated noise)
        if self.cfg["gaussian_process_denoising"]:
            if np.random.rand() < self.cfg["gaussian_process_rate"]:
                factor = self.cfg["gaussian_process_scaling_factor"]
                sigma = self.cfg["gaussian_process_sigma"]
                gp_h = int(h / factor)
                gp_w = int(w / factor)
                gp_noise = np.random.normal(0, sigma,
                                            (gp_h, gp_w)).astype(np.float32)
                gp_noise = cv2.resize(gp_noise, (w, h),
                                      interpolation=cv2.INTER_CUBIC)
                # 0이 아닌 픽셀에만 노이즈 추가
                mask = image > 0
                image[mask] += gp_noise[mask]

        # 3) Symmetrize (회전 + 반전)
        if self.cfg["symmetrize"]:
            # 180도 회전 (50%)
            if np.random.rand() < 0.5:
                center = (w / 2, h / 2)
                rot_mat = cv2.getRotationMatrix2D(center, 180.0, 1.0)
                image = cv2.warpAffine(image, rot_mat, (w, h),
                                       flags=cv2.INTER_NEAREST)
            # 좌우 반전 (50%)
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
            # 상하 반전 (50%)
            if np.random.rand() < 0.5:
                image = np.flipud(image).copy()

        return image


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
            all_paths.append((obj_key, pose_key))

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
        obj_key, pose_key = train_paths[idx]
        group = root[obj_key][pose_key]

        image = np.array(group["image"], dtype=np.float64)
        if image.ndim == 3:
            image = image[:, :, 0]

        if image.shape[0] != im_height or image.shape[1] != im_width:
            image = cv2.resize(image.astype(np.float32),
                               (im_width, im_height),
                               interpolation=cv2.INTER_CUBIC).astype(np.float64)

        im_sum += image.sum()
        im_sq_sum += (image ** 2).sum()
        num_pixels += image.size

        depth = float(np.array(group["depth"]))
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
    criterion = nn.CrossEntropyLoss()

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

    # 시드 설정
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── zarr 열어서 전체 샘플 수 파악 ──
    root = zarr.open(str(args.data), mode="r")
    all_paths = []
    for obj_key in root.keys():
        for pose_key in root[obj_key].keys():
            all_paths.append((obj_key, pose_key))
    num_total = len(all_paths)
    log.info(f"Total samples: {num_total}")

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
    train_ds = DexNetZarrDataset(
        args.data, indices=train_indices,
        im_height=args.im_height, im_width=args.im_width,
        metric_thresh=cfg["metric_thresh"], augment=True, cfg=cfg,
        im_mean=im_mean, im_std=im_std,
        pose_mean=pose_mean, pose_std=pose_std)

    val_ds = DexNetZarrDataset(
        args.data, indices=val_indices,
        im_height=args.im_height, im_width=args.im_width,
        metric_thresh=cfg["metric_thresh"], augment=False, cfg=cfg,
        im_mean=im_mean, im_std=im_std,
        pose_mean=pose_mean, pose_std=pose_std)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train_batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_ds, batch_size=cfg["val_batch_size"],
        shuffle=False, num_workers=2, pin_memory=True)

    # ── 모델 ──
    model = DexNet2(im_height=args.im_height, im_width=args.im_width)
    model.im_mean = im_mean
    model.im_std = im_std
    model.pose_mean = np.array([pose_mean], dtype=np.float32)
    model.pose_std = np.array([pose_std], dtype=np.float32)
    model.to(device)

    # ── Loss / Optimizer / Scheduler ──
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["base_lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"])

    # lr decay: 매 (decay_step_multiplier * num_train) 샘플마다 decay
    decay_step_epochs = cfg["decay_step_multiplier"]
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(decay_step_epochs * len(train_loader))),
        gamma=cfg["decay_rate"])

    # ── 출력 디렉토리 ──
    os.makedirs(args.output, exist_ok=True)

    # ── 학습 ──
    log.info("=" * 60)
    log.info("Starting training")
    log.info("=" * 60)

    global_step = 0
    best_val_error = 1.0

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

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e10)

            optimizer.step()
            scheduler.step()

            # 통계
            preds = logits.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)

            epoch_loss += loss.item() * batch_total
            epoch_correct += batch_correct
            epoch_total += batch_total
            global_step += 1

            if global_step % cfg["log_frequency"] == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info(
                    f"  [epoch {epoch} step {global_step}] "
                    f"loss={loss.item():.4f}  "
                    f"batch_acc={batch_correct/batch_total:.3f}  "
                    f"lr={lr_now:.6f}")

        # ── Epoch 요약 ──
        train_loss = epoch_loss / max(epoch_total, 1)
        train_error = 1.0 - epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch}/{cfg['num_epochs']}  "
            f"train_loss={train_loss:.4f}  train_error={train_error:.4f}  "
            f"time={elapsed:.1f}s")

        # ── Validation ──
        if epoch % cfg["eval_frequency"] == 0 or epoch == cfg["num_epochs"]:
            val_loss, val_error = evaluate(model, val_loader, device)
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
    parser = argparse.ArgumentParser(description="Dex-Net 2.0 GQ-CNN 학습")
    parser.add_argument("--data", type=str, required=True,
                        help="zarr 데이터셋 경로")
    parser.add_argument("--output", type=str, default="./output",
                        help="모델 저장 경로")
    parser.add_argument("--im-height", type=int, default=32,
                        help="이미지 높이 (기본 32)")
    parser.add_argument("--im-width", type=int, default=32,
                        help="이미지 너비 (기본 32)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="에포크 수 (기본 50)")
    parser.add_argument("--lr", type=float, default=None,
                        help="학습률 (기본 0.01)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="배치 크기 (기본 64)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()