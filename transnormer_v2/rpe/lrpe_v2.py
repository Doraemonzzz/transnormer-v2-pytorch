# https://github.com/zh217/torch-dct
# https://github.com/zh217/torch-dct/issues/15
# 还是单头版本, 但是适配多维, only test for householder and rope

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from ..helpers import print_params, logging_info

class LrpeV2(nn.Module):
    def __init__(
        self,
        number_head=8,
        embedding_dim=64,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        # mix transform
        d = number_head * embedding_dim
        # (h, 1, e // 2)
        self.theta = nn.Parameter(
            10000
            ** (-2 / d * torch.arange(d // 2 // 2)).reshape(
                number_head, 1, -1
            )
        )
        self.v = nn.Parameter(torch.randn(number_head, 1, embedding_dim))
        self.p = self.householder
        self.core_transform = self.mix_rope

    def forward(self, x):
        # b, l, e
        x = self.p(x)
        x = self.core_transform(x)
        return x

    def mix_rope(self, x):
        d = x.shape[-1]
        assert d >= 3
        # split
        e = d // 2
        # 转换为偶数
        if e % 2:
            e += 1
        return self.mix_transform(x, e)

    def mix_transform(self, x, e):
        assert e % 2 == 0
        l, d = x.shape[-2], x.shape[-1]
        m = len(x.shape)
        # 后e项
        x1 = x[..., e:]
        # 前e项做rope
        x = x[..., :e]
        theta = self.theta
        theta = torch.stack([theta, theta], dim=-1).reshape(-1, 1, e)
        # l, 1
        index = torch.arange(l).reshape(-1, 1).to(x)
        # ...., l, 1
        for _ in range(len(theta.shape) - 2):
            index = index.unsqueeze(0)
        theta = theta * index
        # (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
        x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x_transform = x * torch.cos(theta) + x_half * torch.sin(theta)

        if e != d:
            x_transform = torch.cat([x_transform, x1], dim=-1)

        return x_transform

    def householder(self, x, eps=1e-6):
        v = self.v / (torch.norm(self.v) + eps)
        # 调整为相同形状
        for i in range(len(x.shape) - 1):
            v = v.unsqueeze(0)
        # b, n, e; 1, 1, e -> 1, n, 1
        y = torch.einsum("...ne,...le->...nl", x, v)
        # 1, n, 1; 1, 1, e -> 1, n, e
        y = torch.einsum("...nl,...le->...ne", y, v)

        return x - 2 * y
