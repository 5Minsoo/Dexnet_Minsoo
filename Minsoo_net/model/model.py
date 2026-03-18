# -*- coding: utf-8 -*-
"""
Dex-Net 2.0 GQ-CNN — PyTorch 구현.

Architecture:
  im_stream:  conv1_1(7,64) → conv1_2(5,64) → conv2_1(3,64) → conv2_2(3,64) → fc3(1024)
  pose_stream: pc1(16)
  merge_stream: fc4_merge(1024) → fc5(2)
Input:  H×W×1 depth image + scalar gripper depth (pose_dim=1)
Output: logits (2) during training, softmax probs during inference
"""
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── LRN 파라미터 ─────────────────────────────────────────────────────
LRN_SIZE = 5
LRN_ALPHA = 2e-05
LRN_BETA = 0.75
LRN_BIAS = 1.0


def _logger(name="DexNet2"):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg


class DexNet2(nn.Module):
    """Dex-Net 2.0 GQ-CNN."""

    def __init__(self, im_height=32, im_width=32, im_channels=1, pose_dim=1):
        super().__init__()
        self.log = _logger()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = im_channels
        self.pose_dim = pose_dim

        # ── Image stream ──
        self.conv1_1 = nn.Conv2d(im_channels, 64, 7, padding=3)
        self.pool1_1 = nn.MaxPool2d(1, stride=1)

        self.conv1_2 = nn.Conv2d(64, 64, 5, padding=2)
        self.pool1_2 = nn.MaxPool2d(2, stride=2)
        self.lrn1_2 = nn.LocalResponseNorm(LRN_SIZE, alpha=LRN_ALPHA,
                                            beta=LRN_BETA, k=LRN_BIAS)

        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2_1 = nn.MaxPool2d(1, stride=1)

        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2_2 = nn.MaxPool2d(2, stride=2)
        self.lrn2_2 = nn.LocalResponseNorm(LRN_SIZE, alpha=LRN_ALPHA,
                                            beta=LRN_BETA, k=LRN_BIAS)

        fc3_in = (im_height // 4) * (im_width // 4) * 64
        self.fc3 = nn.Linear(fc3_in, 1024)

        # ── Pose stream ──
        self.pc1 = nn.Linear(pose_dim, 16)

        # ── Merge stream ──
        self.fc4_w_im = nn.Linear(1024, 1024, bias=False)
        self.fc4_w_pose = nn.Linear(16, 1024, bias=False)
        self.fc4_bias = nn.Parameter(torch.zeros(1024))

        self.fc5 = nn.Linear(1024, 2)
        nn.init.zeros_(self.fc5.bias)

        # ── 정규화 통계 ──
        self.im_mean = 0.0
        self.im_std = 1.0
        self.pose_mean = np.zeros(pose_dim, dtype=np.float32)
        self.pose_std = np.ones(pose_dim, dtype=np.float32)

    def forward(self, im, pose, drop_rate=0.0):
        """
        Returns (B, 2) logits. softmax은 loss 또는 predict에서 처리.
        """
        x = F.relu(self.conv1_1(im))
        x = self.pool1_1(x)

        x = F.relu(self.conv1_2(x))
        x = self.lrn1_2(x)
        x = self.pool1_2(x)

        x = F.relu(self.conv2_1(x))
        x = self.pool2_1(x)

        x = F.relu(self.conv2_2(x))
        x = self.lrn2_2(x)
        x = self.pool2_2(x)

        x = x.flatten(1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=drop_rate, training=self.training)

        p = F.relu(self.pc1(pose))

        m = self.fc4_w_im(x) + self.fc4_w_pose(p) + self.fc4_bias
        m = F.relu(m)
        m = F.dropout(m, p=drop_rate, training=self.training)

        return self.fc5(m)

    @torch.no_grad()
    def predict(self, images, poses):
        """
        images : (N, H, W, 1) raw depth (NHWC)
        poses  : (N, 1) raw gripper depth
        Returns (N, 2) [P(fail), P(success)]
        """
        self.eval()
        dev = next(self.parameters()).device

        im_n = (images - self.im_mean) / self.im_std
        po_n = (poses - self.pose_mean) / self.pose_std

        im_t = torch.from_numpy(im_n.transpose(0, 3, 1, 2).astype(np.float32)).to(dev)
        po_t = torch.from_numpy(po_n.astype(np.float32)).to(dev)

        outs = []
        for i in range(0, im_t.shape[0], 64):
            logits = self(im_t[i:i+64], po_t[i:i+64])
            outs.append(F.softmax(logits, dim=-1))
        return torch.cat(outs, 0).cpu().numpy()

    def predict_success(self, images, poses):
        return self.predict(images, poses)[:, 1]

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "im_mean": self.im_mean, "im_std": self.im_std,
            "pose_mean": self.pose_mean, "pose_std": self.pose_std,
            "im_height": self.im_height, "im_width": self.im_width,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = cls(im_height=ckpt["im_height"], im_width=ckpt["im_width"])
        m.load_state_dict(ckpt["state_dict"])
        m.im_mean = ckpt["im_mean"]
        m.im_std = ckpt["im_std"]
        m.pose_mean = ckpt["pose_mean"]
        m.pose_std = ckpt["pose_std"]
        m.eval()
        return m