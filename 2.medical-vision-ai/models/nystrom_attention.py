# pylint: disable=C0114,C0115,C0116
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from math import ceil


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)

    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        residual_conv_kernel: int = 33,
        eps: float = 1e-8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.eps = eps
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads, heads,
                (residual_conv_kernel, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                return_attn: bool = False, return_attn_matrices: bool = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad sequence for divisibility
        if (padding := n % m) > 0:
            pad_len = m - padding
            x = F.pad(x, (0, 0, pad_len, 0), value=0)
            if exists(mask):
                mask = F.pad(mask, (pad_len, 0), value=False)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        l = ceil(n / m)
        q_landmarks = reduce(q, "... (n l) d -> ... n d", "sum", l=l)
        k_landmarks = reduce(k, "... (n l) d -> ... n d", "sum", l=l)

        divisor = l
        if exists(mask):
            mask_lm_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_lm_sum[..., None] + eps
            q_landmarks /= divisor
            k_landmarks /= divisor
        else:
            q_landmarks /= divisor
            k_landmarks /= divisor

        sim1 = einsum("... i d, ... j d -> ... i j", q, k_landmarks)
        sim2 = einsum("... i d, ... j d -> ... i j", q_landmarks, k_landmarks)
        sim3 = einsum("... i d, ... j d -> ... i j", q_landmarks, k)

        if exists(mask):
            mask_val = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * (mask_lm_sum > 0)[..., None, :]), mask_val)
            sim2.masked_fill_(~((mask_lm_sum > 0)[..., None] * (mask_lm_sum > 0)[..., None, :]), mask_val)
            sim3.masked_fill_(~((mask_lm_sum > 0)[..., None] * mask[..., None, :]), mask_val)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            out = out + self.res_conv(v)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn_matrices:
            return out, (attn1, attn2_inv, attn3)
        if return_attn:
            return out, attn1 @ attn2_inv @ attn3
        return out


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Nystromformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        attn_values_residual: bool = True,
        attn_values_residual_conv_kernel: int = 33,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(
                    dim,
                    NystromAttention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        num_landmarks=num_landmarks,
                        pinv_iterations=pinv_iterations,
                        residual=attn_values_residual,
                        residual_conv_kernel=attn_values_residual_conv_kernel,
                        dropout=attn_dropout,
                    )
                ),
                PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x
