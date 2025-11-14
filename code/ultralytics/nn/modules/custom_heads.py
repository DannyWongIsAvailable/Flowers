# ultralytics/nn/modules/custom_heads.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------
# 1) GeM Pooling
# ----------------------
class GeM(nn.Module):
    """Generalized Mean Pooling (Learnable p)."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        return F.avg_pool2d(
            x.pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


# ----------------------
# 2) ArcMargin (ArcFace)
# ----------------------
class ArcMargin(nn.Module):
    """
    ArcFace / Additive Angular Margin Softmax.
    推理时只输入 logits；训练时需 labels。
    """
    def __init__(self, out_features, s=64.0, m=0.30, easy_margin=False):
        super().__init__()
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # 延迟构建
        self.weight = None

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def build(self, in_features, device):
        """根据输入特征数构建权重，并迁移到设备上"""
        self.weight = nn.Parameter(torch.empty(self.out_features, in_features, device=device))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        if self.weight is None:
            self.build(x.size(1), x.device)

        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)

        logits = torch.mm(x_norm, w_norm.t())

        if labels is None:
            return logits * self.s

        # Add margin
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        sine = torch.sqrt((1.0 - logits ** 2).clamp(0, 1))
        phi = logits * self.cos_m - sine * self.sin_m

        if not self.easy_margin:
            phi = torch.where(logits > self.th, phi, logits - self.mm)

        output = logits * (1 - one_hot) + phi * one_hot
        return output * self.s


# ----------------------
# 3) FlowerHead（自定义分类头）
# ----------------------
class FlowerHead(nn.Module):
    """
    包含 GeM → BN → Dropout → Linear → GELU → BN → Dropout → ArcMargin
    所有层均自动根据输入通道数构建（兼容所有 YOLO scale）
    """
    def __init__(self, nc, p=0.4, m=0.3):
        super().__init__()
        self.nc = nc
        self.p = p
        self.m = m

        # 子模块：第一次 forward 再 build
        self.bn1 = None
        self.dp1 = nn.Dropout(p)
        self.fc = None
        self.act = nn.GELU()
        self.bn2 = None
        self.dp2 = nn.Dropout(p)
        self.arc = None

        self.gem = GeM()

    def build(self, c, device):
        """根据实际输入通道数初始化所有层，并迁移到对应设备"""
        self.bn1 = nn.BatchNorm2d(c).to(device)
        self.fc = nn.Linear(c, c).to(device)
        self.bn2 = nn.BatchNorm1d(c).to(device)
        self.arc = ArcMargin(self.nc, m=self.m).to(device)

    def forward(self, x, labels=None):
        # 1) GeM pooling
        x = self.gem(x)
        x = x.squeeze(-1).squeeze(-1)  # N, C

        # 2) 第一次 forward 自动构建所有层
        if self.bn1 is None:
            self.build(x.size(1), x.device)

        # 3) 分类头前向
        x = self.bn1(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        x = self.dp1(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.bn2(x)
        x = self.dp2(x)

        # 4) ArcMargin
        return self.arc(x, labels)
