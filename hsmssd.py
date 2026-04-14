# models/hsmssd.py

import math
import torch
import torch.nn as nn

class HSMSSD(nn.Module):
    """
    Hidden-State Mixer (HSM-SSD)——完全保留原始算法思路，
    但输入 x 的形状是 (B, C, L)，其中 L=H*W。
    """

    def __init__(self, d_model, ssd_expand=1, A_init_range=(1,16), state_dim=32):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        # 1) 通道投影：Conv1d(d_model -> 3*state_dim)
        self.BCdt_proj = nn.Conv1d(
            in_channels=d_model,
            out_channels=3 * state_dim,
            kernel_size=1,
            bias=True
        )
        # 2) 深度卷积：保持形状 (B,3*state_dim,H,H)
        conv_dim = 3 * state_dim
        self.dw = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=conv_dim,
            bias=True
        )
        # 3) 隐状态前向投影：Conv1d(d_model -> 2*d_inner)
        self.hz_proj = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * self.d_inner,
            kernel_size=1,
            bias=True
        )
        # 4) 隐状态后向投影：Conv1d(d_inner -> d_model)
        self.out_proj = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=d_model,
            kernel_size=1,
            bias=True
        )

        # 可学习的注意力参数 A，门控参数 D
        A = torch.empty(state_dim).uniform_(*A_init_range)
        self.A = nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (B, C, L), L=H*W
        Returns:
            y: Tensor, shape (B, C, H, H)
            h2: Tensor, shape (B, C, state_dim)
        """
        B, C, L = x.shape
        H = int(math.sqrt(L))

        # ——— 1) 通道投影 ———
        # (B, C, L) -> (B, 3*state_dim, L)
        BCdt = self.BCdt_proj(x)

        # ——— 2) 变回空间图、Depthwise Conv ———
        BCdt = BCdt.view(B, 3 * self.state_dim, H, H)
        BCdt = self.dw(BCdt)

        # ——— 3) flatten 回 (B, 3*state_dim, L)，并拆分 ———
        BCdt = BCdt.flatten(2)  # (B, 3*state_dim, L)
        bs, cs, dt = torch.split(BCdt,
                                 [self.state_dim]*3,
                                 dim=1)

        # ——— 4) 计算带参数 A 的注意力 ———
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)  # (B, state_dim, L)
        AB = A * bs  # (B, state_dim, L)

        # ——— 5) 隐状态交互 ———
        # x: (B, C, L), AB.transpose: (B, L, state_dim)
        h = x @ AB.transpose(-2, -1)  # => (B, C, state_dim)

        # ——— 6) Gate 融合 & 投影 ———
        # h: (B, C, state_dim)
        h_proj = self.hz_proj(h)  # (B, 2*d_inner, state_dim)
        h1, z1 = torch.split(h_proj, [self.d_inner, self.d_inner], dim=1)
        h2 = self.out_proj(h1 * self.act(z1) + h1 * self.D)  # (B, C, state_dim)

        # ——— 7) 最终输出 y ———
        # h2: (B, C, state_dim), cs: (B, state_dim, L)
        y = h2 @ cs  # (B, C, L)
        y = y.view(B, C, H, H).contiguous()

        return y, h2
