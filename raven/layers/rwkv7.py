from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _low_rank_slot_rwkv7_fwd_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        slot_ptr,
        out_ptr,
        T: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_s = tl.program_id(2)

        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, RANK)
        offs_g = tl.arange(0, GROUP)
        state = tl.zeros((RANK, D), dtype=tl.float32)

        for t in range(0, T):
            token_base = (pid_b * T + t) * C + pid_h * D
            slot_offset = ((pid_b * T + t) * H + pid_h) * S + pid_s

            rt = tl.load(r_ptr + token_base + offs_d).to(tl.float32)
            wt = tl.load(w_ptr + token_base + offs_d).to(tl.float32)
            kt = tl.load(k_ptr + token_base + offs_d).to(tl.float32)
            vt = tl.load(v_ptr + token_base + offs_d).to(tl.float32)
            at = tl.load(a_ptr + token_base + offs_d).to(tl.float32)
            bt = tl.load(b_ptr + token_base + offs_d).to(tl.float32)
            zt = tl.load(slot_ptr + slot_offset).to(tl.float32)

            grouped = offs_r[:, None] * GROUP + offs_g[None, :]
            k_rank = tl.sum(tl.load(k_ptr + token_base + grouped).to(tl.float32), axis=1) / GROUP
            r_rank = tl.sum(tl.load(r_ptr + token_base + grouped).to(tl.float32), axis=1) / GROUP

            decay = tl.exp(-tl.exp(wt))
            sa = tl.sum(state * at[None, :], axis=1)
            update = k_rank[:, None] * vt[None, :] + sa[:, None] * bt[None, :]
            state = state * (1.0 - zt + zt * decay[None, :]) + zt * update

            yt = tl.sum(state * r_rank[:, None], axis=0) * zt
            tl.atomic_add(out_ptr + token_base + offs_d, yt, sem="relaxed")


def _ortho_init_(x: Tensor, scale: float = 1.0) -> Tensor:
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
            for i in range(shape[0]):
                nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            raise ValueError(f"Unsupported tensor shape for RWKV-7 orthogonal init: {shape}")
    return x


def _time_shift_delta(x: Tensor) -> Tensor:
    xx = torch.empty_like(x)
    xx[:, 0] = -x[:, 0]
    if x.size(1) > 1:
        xx[:, 1:] = x[:, :-1] - x[:, 1:]
    return xx


def rwkv7_recurrence_torch(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    head_size: int,
) -> Tensor:
    B, T, C = r.shape
    H = C // head_size
    dtype = r.dtype
    r = r.reshape(B, T, H, head_size)
    k = k.reshape(B, T, H, head_size)
    v = v.reshape(B, T, H, head_size)
    a = a.reshape(B, T, H, head_size)
    b = b.reshape(B, T, H, head_size)
    decay = torch.exp(-torch.exp(w.float())).reshape(B, T, H, head_size)
    state = torch.zeros(B, H, head_size, head_size, device=r.device, dtype=torch.float32)
    out = []
    for t in range(T):
        rt = r[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        at = a[:, t].float()
        bt = b[:, t].float()
        wt = decay[:, t]
        sa = (state * at.unsqueeze(-2)).sum(dim=-1)
        state = state * wt.unsqueeze(-2) + vt.unsqueeze(-1) * kt.unsqueeze(-2) + sa.unsqueeze(-1) * bt.unsqueeze(-2)
        out.append((state * rt.unsqueeze(-2)).sum(dim=-1).to(dtype))
    return torch.stack(out, dim=1).reshape(B, T, C)


def rwkv7_slot_recurrence_torch(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    slot_weights: Tensor,
    head_size: int,
) -> Tensor:
    B, T, C = r.shape
    H = C // head_size
    S = slot_weights.shape[-1]
    dtype = r.dtype
    r = r.reshape(B, T, H, head_size)
    k = k.reshape(B, T, H, head_size)
    v = v.reshape(B, T, H, head_size)
    a = a.reshape(B, T, H, head_size)
    b = b.reshape(B, T, H, head_size)
    decay = torch.exp(-torch.exp(w.float())).reshape(B, T, H, head_size)
    state = torch.zeros(B, H, S, head_size, head_size, device=r.device, dtype=torch.float32)
    out = []
    for t in range(T):
        rt = r[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        at = a[:, t].float()
        bt = b[:, t].float()
        wt = decay[:, t]
        z = slot_weights[:, t].float()

        sa = (state * at.unsqueeze(2).unsqueeze(-2)).sum(dim=-1)
        update = vt.unsqueeze(2).unsqueeze(-1) * kt.unsqueeze(2).unsqueeze(-2)
        update = update + sa.unsqueeze(-1) * bt.unsqueeze(2).unsqueeze(-2)

        z_state = z.unsqueeze(-1).unsqueeze(-1)
        state = state * (1.0 - z_state + z_state * wt.unsqueeze(2).unsqueeze(-2)) + z_state * update

        slot_y = (state * rt.unsqueeze(2).unsqueeze(-2)).sum(dim=-1)
        yt = (slot_y * z.unsqueeze(-1)).sum(dim=2)
        out.append(yt.to(dtype))
    return torch.stack(out, dim=1).reshape(B, T, C)


def rwkv7_low_rank_slot_recurrence_torch(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    slot_weights: Tensor,
    head_size: int,
    rank: int,
) -> Tensor:
    B, T, C = r.shape
    H = C // head_size
    S = slot_weights.shape[-1]
    dtype = r.dtype
    r = r.reshape(B, T, H, head_size)
    k = k.reshape(B, T, H, head_size)
    v = v.reshape(B, T, H, head_size)
    a = a.reshape(B, T, H, head_size)
    b = b.reshape(B, T, H, head_size)
    decay = torch.exp(-torch.exp(w.float())).reshape(B, T, H, head_size)

    state = torch.zeros(B, H, S, rank, head_size, device=r.device, dtype=torch.float32)
    out = []
    for t in range(T):
        rt = r[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        at = a[:, t].float()
        bt = b[:, t].float()
        wt = decay[:, t]
        z = slot_weights[:, t].float()

        k_rank = kt.reshape(B, H, rank, head_size // rank).mean(dim=-1)
        r_rank = rt.reshape(B, H, rank, head_size // rank).mean(dim=-1)

        sa = (state * at.unsqueeze(2).unsqueeze(2)).sum(dim=-1)
        update = k_rank.unsqueeze(2).unsqueeze(-1) * vt.unsqueeze(2).unsqueeze(3)
        update = update + sa.unsqueeze(-1) * bt.unsqueeze(2).unsqueeze(3)

        z_state = z.unsqueeze(-1).unsqueeze(-1)
        state = state * (1.0 - z_state + z_state * wt.unsqueeze(2).unsqueeze(2)) + z_state * update

        slot_y = (state * r_rank.unsqueeze(2).unsqueeze(-1)).sum(dim=3)
        yt = (slot_y * z.unsqueeze(-1)).sum(dim=2)
        out.append(yt.to(dtype))
    return torch.stack(out, dim=1).reshape(B, T, C)


def rwkv7_low_rank_slot_recurrence_triton(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    slot_weights: Tensor,
    head_size: int,
    rank: int,
) -> Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not r.is_cuda:
        raise RuntimeError("Triton low-rank slot RWKV requires CUDA tensors")
    if r.dtype != torch.bfloat16:
        raise RuntimeError("Triton low-rank slot RWKV currently requires bfloat16 tensors")
    if head_size % rank != 0:
        raise ValueError(f"head_size ({head_size}) must be divisible by rank ({rank})")

    B, T, C = r.shape
    H = C // head_size
    S = slot_weights.shape[-1]
    group = head_size // rank
    out = torch.zeros_like(r)
    _low_rank_slot_rwkv7_fwd_kernel[(B, H, S)](
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        slot_weights.contiguous(),
        out,
        T,
        C,
        H,
        S,
        head_size,
        rank,
        group,
        num_warps=4,
    )
    return out


class _LowRankSlotRWKV7TritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, slot_weights: Tensor, head_size: int, rank: int):
        ctx.head_size = head_size
        ctx.rank = rank
        ctx.save_for_backward(r, w, k, v, a, b, slot_weights)
        return rwkv7_low_rank_slot_recurrence_triton(r, w, k, v, a, b, slot_weights, head_size, rank)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        saved = ctx.saved_tensors
        with torch.enable_grad():
            r, w, k, v, a, b, slot_weights = [tensor.detach().requires_grad_(True) for tensor in saved]
            y = rwkv7_low_rank_slot_recurrence_torch(
                r,
                w,
                k,
                v,
                a,
                b,
                slot_weights,
                ctx.head_size,
                ctx.rank,
            )
        grads = torch.autograd.grad(
            y,
            (r, w, k, v, a, b, slot_weights),
            grad_out,
            allow_unused=True,
        )
        return (*grads, None, None)


def rwkv7_low_rank_slot_recurrence_triton_autograd(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    slot_weights: Tensor,
    head_size: int,
    rank: int,
) -> Tensor:
    return _LowRankSlotRWKV7TritonFn.apply(r, w, k, v, a, b, slot_weights, head_size, rank)


@dataclass
class RWKV7Config:
    max_seq_len: int
    n_layers: int
    hidden_size: int
    norm_eps: float = 1e-6
    rwkv7_head_size: int = 64
    rwkv7_backend: str = "auto"
    rwkv7_chunk_len: int = 16
    rwkv7_enable_v_first_mix: bool = True

    @property
    def init_std(self) -> float:
        return self.hidden_size ** -0.5


class RWKV7TimeMix(nn.Module):
    """RWKV-7 time mixer adapted from HRM-Text's original RWKV integration."""

    def __init__(self, config: RWKV7Config, layer_id: int) -> None:
        super().__init__()
        dim = config.hidden_size
        head_size = config.rwkv7_head_size
        if dim % head_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by rwkv7_head_size ({head_size})")
        if config.rwkv7_backend not in {"auto", "cuda", "torch"}:
            raise ValueError(f"Unknown RWKV-7 backend: {config.rwkv7_backend}")

        self.dim = dim
        self.depth = config.n_layers
        self.layer_id = layer_id
        self.head_size = head_size
        self.n_head = dim // head_size
        self.backend = config.rwkv7_backend
        self.chunk_len = config.rwkv7_chunk_len
        self.enable_v_first_mix = config.rwkv7_enable_v_first_mix
        self.init_std = config.init_std

        decay_lora_dim = max(32, int(round((2.5 * (dim**0.5)) / 32) * 32))
        aaa_lora_dim = max(32, int(round((2.5 * (dim**0.5)) / 32) * 32))
        gate_lora_dim = max(32, int(round((5.0 * (dim**0.5)) / 32) * 32))
        mv_lora_dim = max(32, int(round((1.7 * (dim**0.5)) / 32) * 32))

        self.x_r = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_w = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_k = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_v = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_a = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_g = nn.Parameter(torch.empty(dim, dtype=torch.float32))

        self.w1 = nn.Parameter(torch.empty(dim, decay_lora_dim, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.empty(decay_lora_dim, dim, dtype=torch.float32))
        self.w0 = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.a1 = nn.Parameter(torch.empty(dim, aaa_lora_dim, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.empty(aaa_lora_dim, dim, dtype=torch.float32))
        self.a0 = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        if self.enable_v_first_mix:
            self.v1 = nn.Parameter(torch.empty(dim, mv_lora_dim, dtype=torch.float32))
            self.v2 = nn.Parameter(torch.empty(mv_lora_dim, dim, dtype=torch.float32))
            self.v0 = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.g1 = nn.Parameter(torch.empty(dim, gate_lora_dim, dtype=torch.float32))
        self.g2 = nn.Parameter(torch.empty(gate_lora_dim, dim, dtype=torch.float32))
        self.k_k = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.k_a = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.r_k = nn.Parameter(torch.empty(self.n_head, head_size, dtype=torch.float32))

        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, dim, eps=64e-5)
        self.reset_parameters()

    def reset_parameters(self, init_std: Optional[float] = None) -> None:
        init_std = self.init_std if init_std is None else init_std
        device = self.x_r.device
        dim = self.dim
        ratio_0_to_1 = self.layer_id / max(self.depth - 1, 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_id / max(self.depth, 1))
        ddd = torch.arange(dim, device=device, dtype=torch.float32) / dim
        linear = torch.arange(dim, device=device, dtype=torch.float32) / max(dim - 1, 1) - 0.5
        zigzag = torch.arange(dim, device=device, dtype=torch.float32) % self.head_size
        zigzag = (zigzag - ((self.head_size - 1) / 2)) / max((self.head_size - 1) / 2, 1.0)
        zigzag = zigzag * zigzag.abs()
        decay = -6 + 6 * (torch.arange(dim, device=device, dtype=torch.float32) / max(dim - 1, 1)) ** (
            1 + ratio_0_to_1**0.3
        )
        with torch.no_grad():
            self.x_r.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.w0.copy_(decay + 0.5 + zigzag * 2.5)
            self.a0.copy_(torch.zeros_like(linear) - 0.19 + zigzag * 0.3 + linear * 0.4)
            if self.enable_v_first_mix:
                self.v0.copy_(torch.zeros_like(linear) + 0.73 - linear * 0.4)
            self.k_k.copy_(torch.zeros_like(linear) + 0.71 - linear * 0.1)
            self.k_a.fill_(1.02)
            self.r_k.fill_(-0.04)
            self.w1.zero_()
            _ortho_init_(self.w2, 0.1)
            self.a1.zero_()
            _ortho_init_(self.a2, 0.1)
            if self.enable_v_first_mix:
                self.v1.zero_()
                _ortho_init_(self.v2, 0.1)
            self.g1.zero_()
            _ortho_init_(self.g2, 0.1)
        for proj in (self.receptance, self.key, self.value):
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)
        nn.init.zeros_(self.output.weight)
        self.ln_x.reset_parameters()

    def _can_use_cuda(self, x: Tensor) -> bool:
        return self.backend != "torch" and x.is_cuda and x.dtype == torch.bfloat16 and self.head_size == 64

    def _validate_cuda_required(self, x: Tensor) -> None:
        if self.backend != "cuda":
            return
        if not x.is_cuda:
            raise RuntimeError("RWKV-7 backend='cuda' requires CUDA input tensors.")
        if x.dtype != torch.bfloat16:
            raise RuntimeError("RWKV-7 backend='cuda' requires bfloat16 input tensors.")
        if self.head_size != 64:
            raise RuntimeError("RWKV-7 backend='cuda' currently requires head_size=64 for the LT2 kernels.")
        if self.chunk_len != 16:
            raise RuntimeError("RWKV-7 backend='cuda' currently requires chunk_len=16 for the LT2 kernels.")

    def _forward_cuda(self, x: Tensor, v_first: Optional[Tensor], reset_v_first: bool) -> Tuple[Tensor, Tensor]:
        try:
            from apps.LT2 import rwkv7_cuda
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RWKV-7 CUDA backend requires the LT2 wrapper package on PYTHONPATH. "
                "For this workspace, run with `PYTHONPATH=/home/xiaol/X/HRM-Text:$PYTHONPATH` "
                "or pass `--lt2-wrapper-root /home/xiaol/X/HRM-Text` to examples/compare_mixers.py."
            ) from exc

        x = x.contiguous()
        xr, xw, xk, xv, xa, xg = rwkv7_cuda.tmix_mix6(
            x,
            self.x_r.to(device=x.device, dtype=x.dtype),
            self.x_w.to(device=x.device, dtype=x.dtype),
            self.x_k.to(device=x.device, dtype=x.dtype),
            self.x_v.to(device=x.device, dtype=x.dtype),
            self.x_a.to(device=x.device, dtype=x.dtype),
            self.x_g.to(device=x.device, dtype=x.dtype),
        )
        r = self.receptance(xr)
        w = self.w0.to(dtype=x.dtype).view(1, 1, -1) + (
            torch.tanh(xw @ self.w1.to(dtype=x.dtype)) @ self.w2.to(dtype=x.dtype)
        )
        k = self.key(xk)
        v = self.value(xv)
        if reset_v_first or v_first is None:
            v_first = v
        elif self.enable_v_first_mix:
            v12 = (xv @ self.v1.to(dtype=x.dtype)) @ self.v2.to(dtype=x.dtype)
            v = rwkv7_cuda.tmix_vres_gate(v, v_first, self.v0.to(device=x.device, dtype=x.dtype), v12)
        a = rwkv7_cuda.tmix_a_gate(
            self.a0.to(device=x.device, dtype=x.dtype),
            (xa @ self.a1.to(dtype=x.dtype)) @ self.a2.to(dtype=x.dtype),
        )
        g = torch.sigmoid(xg @ self.g1.to(dtype=x.dtype)) @ self.g2.to(dtype=x.dtype)
        k, neg_kk, kka = rwkv7_cuda.tmix_kk_pre(
            k,
            self.k_k.to(device=x.device, dtype=x.dtype),
            a,
            self.k_a.to(device=x.device, dtype=x.dtype),
            self.head_size,
        )
        y = rwkv7_cuda.rwkv7_recurrence_cuda_bf16(r, w, k, v, neg_kk, kka, self.head_size, self.chunk_len)
        y = rwkv7_cuda.tmix_lnx_rkvres_xg(
            y.contiguous(),
            r.contiguous(),
            k.contiguous(),
            v.contiguous(),
            self.r_k.to(device=x.device, dtype=x.dtype),
            self.ln_x.weight.to(device=x.device, dtype=x.dtype),
            self.ln_x.bias.to(device=x.device, dtype=x.dtype),
            g.contiguous(),
        )
        return self.output(y), v_first

    def _project_torch(
        self,
        x: Tensor,
        v_first: Optional[Tensor],
        reset_v_first: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, T, C = x.shape
        xx = _time_shift_delta(x)
        xr = x + xx * self.x_r.to(dtype=x.dtype).view(1, 1, -1)
        xw = x + xx * self.x_w.to(dtype=x.dtype).view(1, 1, -1)
        xk = x + xx * self.x_k.to(dtype=x.dtype).view(1, 1, -1)
        xv = x + xx * self.x_v.to(dtype=x.dtype).view(1, 1, -1)
        xa = x + xx * self.x_a.to(dtype=x.dtype).view(1, 1, -1)
        xg = x + xx * self.x_g.to(dtype=x.dtype).view(1, 1, -1)

        r = self.receptance(xr)
        w = self.w0.to(dtype=x.dtype).view(1, 1, -1) + (
            torch.tanh(xw @ self.w1.to(dtype=x.dtype)) @ self.w2.to(dtype=x.dtype)
        )
        k = self.key(xk)
        v = self.value(xv)
        if reset_v_first or v_first is None:
            v_first = v
        elif self.enable_v_first_mix:
            v_mix = torch.sigmoid(
                self.v0.to(dtype=x.dtype).view(1, 1, -1) + (xv @ self.v1.to(dtype=x.dtype)) @ self.v2.to(dtype=x.dtype)
            )
            v = v + (v_first - v) * v_mix
        w = -F.softplus(-w.float()).to(dtype=x.dtype) - 0.5
        a = torch.sigmoid(
            self.a0.to(dtype=x.dtype).view(1, 1, -1) + (xa @ self.a1.to(dtype=x.dtype)) @ self.a2.to(dtype=x.dtype)
        )
        g = torch.sigmoid(xg @ self.g1.to(dtype=x.dtype)) @ self.g2.to(dtype=x.dtype)
        kk = k * self.k_k.to(dtype=x.dtype).view(1, 1, -1)
        kk = F.normalize(kk.reshape(B, T, self.n_head, self.head_size), dim=-1, p=2.0).reshape(B, T, C)
        k = k * (1 + (a - 1) * self.k_a.to(dtype=x.dtype).view(1, 1, -1))
        return r, w, k, v, -kk, kk * a, g, v_first

    def _finish_torch(self, y: Tensor, r: Tensor, k: Tensor, v: Tensor, g: Tensor) -> Tensor:
        B, T, C = y.shape
        y = self.ln_x(y.reshape(B * T, C)).reshape(B, T, C)
        y = y + (
            (
                r.reshape(B, T, self.n_head, self.head_size)
                * k.reshape(B, T, self.n_head, self.head_size)
                * self.r_k.to(dtype=y.dtype)
            ).sum(dim=-1, keepdim=True)
            * v.reshape(B, T, self.n_head, self.head_size)
        ).reshape(B, T, C)
        return self.output(y * g)

    def forward(
        self,
        x: Tensor,
        v_first: Optional[Tensor] = None,
        reset_v_first: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._validate_cuda_required(x)
        if self._can_use_cuda(x):
            try:
                return self._forward_cuda(x, v_first, reset_v_first)
            except Exception as exc:
                if self.backend == "cuda":
                    raise
                warnings.warn(f"Falling back to PyTorch RWKV-7 backend: {exc}", RuntimeWarning, stacklevel=2)

        r, w, k, v, a, b, g, v_first = self._project_torch(x, v_first, reset_v_first)
        y = rwkv7_recurrence_torch(r, w, k, v, a, b, self.head_size)
        return self._finish_torch(y, r, k, v, g), v_first


class RWKV7Mixer(nn.Module):
    """RavenAttention-compatible wrapper around the original RWKV-7 time mixer."""

    accepts_rwkv7_state = True

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        max_seq_len: int,
        layer_idx: int,
        head_size: int = 64,
        backend: str = "auto",
        chunk_len: int = 16,
        enable_v_first_mix: bool = True,
        norm_eps: float = 1e-6,
        **_kwargs,
    ) -> None:
        super().__init__()
        config = RWKV7Config(
            max_seq_len=max_seq_len,
            n_layers=num_hidden_layers,
            hidden_size=hidden_size,
            norm_eps=norm_eps,
            rwkv7_head_size=head_size,
            rwkv7_backend=backend,
            rwkv7_chunk_len=chunk_len,
            rwkv7_enable_v_first_mix=enable_v_first_mix,
        )
        self.layer_idx = layer_idx
        self.time_mix = RWKV7TimeMix(config, layer_idx)

    def reset_parameters(self) -> None:
        self.time_mix.reset_parameters()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        rwkv7_v_first: Optional[Tensor] = None,
        reset_rwkv7_v_first: bool = False,
        **_kwargs,
    ):
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("RWKV7Mixer expects attention_mask with shape [batch_size, seq_len]")
            mask = attention_mask[:, -hidden_states.shape[1] :].to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = hidden_states * mask.unsqueeze(-1)
        else:
            mask = None

        if use_cache:
            warnings.warn(
                "RWKV7Mixer does not maintain an incremental generation cache; falling back to full-prefix execution.",
                RuntimeWarning,
                stacklevel=2,
            )

        hidden_states, rwkv7_v_first = self.time_mix(
            hidden_states,
            v_first=rwkv7_v_first,
            reset_v_first=reset_rwkv7_v_first or rwkv7_v_first is None,
        )

        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1)
            if rwkv7_v_first is not None:
                rwkv7_v_first = rwkv7_v_first * mask.unsqueeze(-1)

        return hidden_states, None, past_key_values, rwkv7_v_first


class RoutedRWKV7Mixer(RWKV7Mixer):
    """RWKV-7 mixer with Raven-style top-k routing over channel slots.

    RWKV-7's recurrent state is dense rather than an explicit slot matrix. This
    adapter gives it a routing-memory axis by partitioning each head into slots
    and using a learned per-token router to gate selected slot groups before and
    after the recurrent update.
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        max_seq_len: int,
        layer_idx: int,
        head_size: int = 64,
        backend: str = "auto",
        chunk_len: int = 16,
        enable_v_first_mix: bool = True,
        norm_eps: float = 1e-6,
        num_slots: int = 64,
        topk: int = 32,
        router_type: str = "lin",
        router_score: str = "sigmoid",
        add_gumbel_noise: bool = True,
        route_floor: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            head_size=head_size,
            backend=backend,
            chunk_len=chunk_len,
            enable_v_first_mix=enable_v_first_mix,
            norm_eps=norm_eps,
            **kwargs,
        )
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if topk <= 0:
            raise ValueError("topk must be positive")
        if router_type not in {"lin", "mlp"}:
            raise ValueError("router_type must be 'lin' or 'mlp'")
        if router_score not in {"sigmoid", "softmax"}:
            raise ValueError("router_score must be 'sigmoid' or 'softmax'")
        if not 0.0 <= route_floor <= 1.0:
            raise ValueError("route_floor must be in [0, 1]")

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = hidden_size // head_size
        self.num_slots = num_slots
        self.topk = min(topk, num_slots)
        self.router_score = router_score
        self.add_gumbel_noise = add_gumbel_noise
        self.route_floor = route_floor

        if router_type == "lin":
            self.router = nn.Linear(hidden_size, self.num_heads * num_slots, bias=False)
        else:
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.GELU(),
                nn.Linear(hidden_size, self.num_heads * num_slots, bias=False),
            )

        slot_ids = torch.arange(head_size, dtype=torch.long) * num_slots // head_size
        self.register_buffer("channel_slot_ids", slot_ids, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if isinstance(self.router, nn.Linear):
            nn.init.zeros_(self.router.weight)
        else:
            first = self.router[0]
            last = self.router[-1]
            nn.init.normal_(first.weight, mean=0.0, std=self.hidden_size ** -0.5)
            nn.init.zeros_(first.bias)
            nn.init.zeros_(last.weight)

    def _route_mask(self, hidden_states: Tensor) -> Tensor:
        B, T, _C = hidden_states.shape
        logits = self.router(hidden_states).view(B, T, self.num_heads, self.num_slots)
        if self.add_gumbel_noise and self.training:
            logits = logits - torch.empty_like(logits).exponential_().log()

        if self.router_score == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)

        route_idx = scores.topk(self.topk, dim=-1).indices
        route_weights = torch.gather(scores, dim=-1, index=route_idx)
        if self.router_score == "sigmoid":
            route_weights = route_weights / (route_weights.sum(dim=-1, keepdim=True) + 1e-9)

        slot_weights = torch.zeros_like(scores).scatter_(-1, route_idx, route_weights)
        channel_weights = torch.gather(
            slot_weights,
            dim=-1,
            index=self.channel_slot_ids.view(1, 1, 1, self.head_size).expand(B, T, self.num_heads, self.head_size),
        )
        scale = self.num_slots / self.topk
        channel_weights = self.route_floor + (1.0 - self.route_floor) * channel_weights * scale
        return channel_weights.reshape(B, T, self.hidden_size).to(dtype=hidden_states.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        rwkv7_v_first: Optional[Tensor] = None,
        reset_rwkv7_v_first: bool = False,
        **kwargs,
    ):
        route_mask = self._route_mask(hidden_states)
        routed_hidden = hidden_states * route_mask
        out, attentions, past_key_values, rwkv7_v_first = super().forward(
            hidden_states=routed_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            rwkv7_v_first=rwkv7_v_first,
            reset_rwkv7_v_first=reset_rwkv7_v_first,
            **kwargs,
        )
        return out * route_mask, attentions, past_key_values, rwkv7_v_first


class SlotRWKV7Mixer(RoutedRWKV7Mixer):
    """RWKV-7 with explicit Raven-like routed recurrent memory slots.

    This variant gives each head `num_slots` independent RWKV state matrices and
    applies the top-k router inside the recurrent update. It is the closest fit
    to Raven's slot-level memory semantics, but it cannot use the dense LT2 RWKV
    recurrence kernel because the kernel has no slot dimension.
    """

    def _slot_weights(self, hidden_states: Tensor) -> Tensor:
        B, T, _C = hidden_states.shape
        logits = self.router(hidden_states).view(B, T, self.num_heads, self.num_slots)
        if self.add_gumbel_noise and self.training:
            logits = logits - torch.empty_like(logits).exponential_().log()

        if self.router_score == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)

        route_idx = scores.topk(self.topk, dim=-1).indices
        route_weights = torch.gather(scores, dim=-1, index=route_idx)
        if self.router_score == "sigmoid":
            route_weights = route_weights / (route_weights.sum(dim=-1, keepdim=True) + 1e-9)
        return torch.zeros_like(scores).scatter_(-1, route_idx, route_weights)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        rwkv7_v_first: Optional[Tensor] = None,
        reset_rwkv7_v_first: bool = False,
        **_kwargs,
    ):
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("SlotRWKV7Mixer expects attention_mask with shape [batch_size, seq_len]")
            mask = attention_mask[:, -hidden_states.shape[1] :].to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = hidden_states * mask.unsqueeze(-1)
        else:
            mask = None

        if use_cache:
            warnings.warn(
                "SlotRWKV7Mixer does not maintain an incremental generation cache; falling back to full-prefix execution.",
                RuntimeWarning,
                stacklevel=2,
            )

        slot_weights = self._slot_weights(hidden_states)
        r, w, k, v, a, b, g, rwkv7_v_first = self.time_mix._project_torch(
            hidden_states,
            v_first=rwkv7_v_first,
            reset_v_first=reset_rwkv7_v_first or rwkv7_v_first is None,
        )
        y = rwkv7_slot_recurrence_torch(r, w, k, v, a, b, slot_weights, self.head_size)
        out = self.time_mix._finish_torch(y, r, k, v, g)

        if mask is not None:
            out = out * mask.unsqueeze(-1)
            if rwkv7_v_first is not None:
                rwkv7_v_first = rwkv7_v_first * mask.unsqueeze(-1)

        return out, None, past_key_values, rwkv7_v_first


class LowRankSlotRWKV7Mixer(SlotRWKV7Mixer):
    """Explicit routed RWKV slots with low-rank per-slot state.

    Full slot RWKV stores `state[slot, D, D]`. This variant stores
    `state[slot, rank, D]`, using grouped averages of the RWKV key/read vectors
    as low-rank write/read coefficients. It keeps slot-level routing while
    making memory and compute scale with `rank * D` instead of `D * D`.
    """

    def __init__(
        self,
        *args,
        low_rank: int = 8,
        low_rank_backend: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if low_rank <= 0:
            raise ValueError("low_rank must be positive")
        if self.head_size % low_rank != 0:
            raise ValueError(f"head_size ({self.head_size}) must be divisible by low_rank ({low_rank})")
        if low_rank_backend not in {"auto", "triton", "triton_autograd", "torch"}:
            raise ValueError("low_rank_backend must be 'auto', 'triton', 'triton_autograd', or 'torch'")
        self.low_rank = low_rank
        self.low_rank_backend = low_rank_backend

    def _should_use_triton(self, r: Tensor) -> bool:
        if self.low_rank_backend == "torch":
            return False
        if self.low_rank_backend in {"triton", "triton_autograd"}:
            return True
        return not torch.is_grad_enabled() and r.is_cuda and r.dtype == torch.bfloat16 and triton is not None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        rwkv7_v_first: Optional[Tensor] = None,
        reset_rwkv7_v_first: bool = False,
        **_kwargs,
    ):
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("LowRankSlotRWKV7Mixer expects attention_mask with shape [batch_size, seq_len]")
            mask = attention_mask[:, -hidden_states.shape[1] :].to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = hidden_states * mask.unsqueeze(-1)
        else:
            mask = None

        if use_cache:
            warnings.warn(
                "LowRankSlotRWKV7Mixer does not maintain an incremental generation cache; falling back to full-prefix execution.",
                RuntimeWarning,
                stacklevel=2,
            )

        slot_weights = self._slot_weights(hidden_states)
        r, w, k, v, a, b, g, rwkv7_v_first = self.time_mix._project_torch(
            hidden_states,
            v_first=rwkv7_v_first,
            reset_v_first=reset_rwkv7_v_first or rwkv7_v_first is None,
        )
        if self._should_use_triton(r):
            triton_fn = (
                rwkv7_low_rank_slot_recurrence_triton_autograd
                if self.low_rank_backend == "triton_autograd" and torch.is_grad_enabled()
                else rwkv7_low_rank_slot_recurrence_triton
            )
            y = triton_fn(
                r,
                w,
                k,
                v,
                a,
                b,
                slot_weights,
                self.head_size,
                self.low_rank,
            )
        else:
            y = rwkv7_low_rank_slot_recurrence_torch(
                r,
                w,
                k,
                v,
                a,
                b,
                slot_weights,
                self.head_size,
                self.low_rank,
            )
        out = self.time_mix._finish_torch(y, r, k, v, g)

        if mask is not None:
            out = out * mask.unsqueeze(-1)
            if rwkv7_v_first is not None:
                rwkv7_v_first = rwkv7_v_first * mask.unsqueeze(-1)

        return out, None, past_key_values, rwkv7_v_first
