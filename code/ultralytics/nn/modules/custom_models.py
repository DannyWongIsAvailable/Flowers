# ultralytics/nn/modules/custom_models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

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


class ChannelAttention(nn.Module):
    """通道注意力模块 - 强化颜色特征"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力模块 - 聚焦花朵主体"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class FlowerAttention(nn.Module):
    """
    花卉专用注意力模块
    结合通道注意力（颜色）和空间注意力（形态）
    适用于细粒度花卉分类
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)  # 先增强颜色特征
        x = self.spatial_att(x)  # 再聚焦空间位置
        return x


class MultiScaleFlowerBlock(nn.Module):
    """
    多尺度花卉特征提取块
    通过不同感受野捕捉从花瓣纹理到整体形态的多层次特征
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # 三个并行分支：小、中、大感受野
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 conv

        # 小尺度分支 - 捕捉细节纹理（花瓣边缘、纹路）
        self.cv2_small = nn.Sequential(
            Conv(c_, c_, 3, 1, g=g),  # 3x3 conv
        )

        # 中尺度分支 - 捕捉局部结构（单个花瓣）
        self.cv2_medium = nn.Sequential(
            Conv(c_, c_, 3, 1, g=g),
            Conv(c_, c_, 3, 1, g=g),  # 两层3x3等效5x5
        )

        # 大尺度分支 - 捕捉整体形态（花朵整体）
        self.cv2_large = nn.Sequential(
            Conv(c_, c_, 3, 1, g=g),
            Conv(c_, c_, 3, 1, g=g),
            Conv(c_, c_, 3, 1, g=g),  # 三层3x3等效7x7
        )

        # 融合不同尺度特征
        self.cv3 = Conv(c_ * 3, c2, 1, 1)

        # 花卉注意力
        self.attention = FlowerAttention(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)

        # 并行提取多尺度特征
        small = self.cv2_small(y)
        medium = self.cv2_medium(y)
        large = self.cv2_large(y)

        # 拼接并融合
        y = torch.cat([small, medium, large], dim=1)
        y = self.cv3(y)

        # 应用注意力
        y = self.attention(y)

        return x + y if self.add else y


class ColorEnhancedConv(nn.Module):
    """
    颜色增强卷积
    针对花卉颜色特征进行专门优化
    对RGB三通道进行差异化处理
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g)

        # 颜色特征增强：为RGB三通道学习不同权重
        if c1 >= 3:
            self.color_enhance = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, c1, 1, groups=c1, bias=False),  # 逐通道权重
                nn.Sigmoid()
            )
        else:
            self.color_enhance = None

    def forward(self, x):
        if self.color_enhance is not None:
            color_weight = self.color_enhance(x)
            x = x * color_weight
        return self.conv(x)


class FlowerC3k2(nn.Module):
    """
    基于C3k2的花卉优化版本
    在原有C3k2基础上添加花卉注意力机制
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        from.block import Bottleneck

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # 使用标准Bottleneck
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))

        # 添加花卉注意力
        self.attention = FlowerAttention(c2)

    def forward(self, x):
        y = torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1)
        y = self.cv3(y)
        y = self.attention(y)
        return y


class FlowerClassifyHead(nn.Module):
    """
    花卉分类专用头部
    增加特征判别性，适配细粒度分类
    """

    def __init__(self, c1, nc):
        super().__init__()

        # 全局平均池化 + 最大池化
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        # 特征融合
        self.fc1 = nn.Sequential(
            nn.Linear(c1 * 2, c1),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # 分类层
        self.fc2 = nn.Linear(c1, nc)

    def forward(self, x):
        # 双池化
        avg = self.pool_avg(x).flatten(1)
        max_out = self.pool_max(x).flatten(1)

        # 融合
        x = torch.cat([avg, max_out], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ============= 用于YAML配置的简化包装 =============

class FlowerAttentionWrapper(nn.Module):
    """YAML配置包装器"""

    def __init__(self, c1, c2, *args):
        super().__init__()
        self.attention = FlowerAttention(c2)
        if c1 != c2:
            self.adapt = Conv(c1, c2, 1)
        else:
            self.adapt = nn.Identity()

    def forward(self, x):
        return self.attention(self.adapt(x))


class MultiScaleFlowerBlockWrapper(nn.Module):
    """YAML配置包装器"""

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.block = MultiScaleFlowerBlock(c1, c2, n, shortcut)

    def forward(self, x):
        return self.block(x)
