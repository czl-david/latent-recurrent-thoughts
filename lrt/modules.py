"""The trainable modules: the proposer g_psi and the refiner r_phi.

Both work in the small dimension d' and share one block class, a bidirectional transformer block
(self-attention with RoPE + SwiGLU); the refiner's transition block f is a separate instance with
independent weights. Every architectural choice below is a key under `model:` in the config.

Proposer:  L0 = P_up(read_K(Blocks([P_down(E[x]) * sqrt(d'); q_1..q_K])))            in R^{K x d}
Refiner:   u = P'_down(L0) * sqrt(d');  z_L, z_H <- z_L^0, z_H^0
           S x H cycles of { T x [z_L <- f(z_L, z_H + u)];  z_H <- f(z_H, z_L) },
           every cycle but the last under stop-gradient;
           Delta = P'_up(z_H);  L* = L0 + Delta
"""

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor: torch.Tensor, std: float, lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    """Truncated normal whose standard deviation AFTER truncation to [lower, upper] std is `std`."""
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()
        sqrt2 = math.sqrt(2)
        a, b = math.erf(lower / sqrt2), math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u, pdf_l = c * math.exp(-0.5 * lower ** 2), c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        tensor.uniform_(a, b).erfinv_().mul_(sqrt2 * comp_std).clip_(lower * comp_std, upper * comp_std)
    return tensor


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.eps) * self.weight.to(x.dtype)


class Rotary(nn.Module):
    """RoPE tables for arbitrary per-example positions."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float().unsqueeze(-1) * self.inv_freq                  # [B, S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: [B, S, H, D]; cos, sin: [B, S, D]."""
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    dtype = q.dtype
    q, k = q.float(), k.float()
    return ((q * cos) + (_rotate_half(q) * sin)).to(dtype), ((k * cos) + (_rotate_half(k) * sin)).to(dtype)


class Attention(nn.Module):
    """Bidirectional multi-head self-attention with an optional key-padding mask."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        trunc_normal_(self.qkv.weight, std=1.0 / math.sqrt(d))
        trunc_normal_(self.o.weight, std=1.0 / math.sqrt(d))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, self.head_dim).unbind(dim=2)
        q, k = apply_rope(q, k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                          # [B, H, S, D]
        mask = None if key_mask is None else key_mask[:, None, None, :]           # True = attend
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(b, s, d))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        trunc_normal_(self.gate_up.weight, std=1.0 / math.sqrt(d))
        trunc_normal_(self.down.weight, std=1.0 / math.sqrt(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    """Self-attention + SwiGLU. `cond` is the recurrence input of the refiner's f(z, c).

    norm='pre':  h = z + Attn(Norm(z + c));  out = h + SwiGLU(Norm(h))
                 The conditioning enters the normalized sublayer input only: adding it to the residual
                 stream as well would let z_H grow geometrically across cycles (z_H + z_L with z_L
                 carrying T copies of z_H + u), since nothing renormalizes the stream.
    norm='post': h = RMSNorm(z + c + Attn(z + c));  out = RMSNorm(h + SwiGLU(h))   (parameter-free norms)
    """

    def __init__(self, d: int, heads: int, mlp_hidden: int, eps: float, norm: str):
        super().__init__()
        assert norm in ("pre", "post"), norm
        self.norm, self.eps = norm, eps
        self.attn = Attention(d, heads)
        self.mlp = SwiGLU(d, mlp_hidden)
        if norm == "pre":
            self.norm1, self.norm2 = RMSNorm(d, eps), RMSNorm(d, eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                key_mask: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        a_in = x if cond is None else x + cond
        if self.norm == "post":
            h = rms_norm(a_in + self.attn(a_in, cos, sin, key_mask), self.eps)
            return rms_norm(h + self.mlp(h), self.eps)
        h = x + self.attn(self.norm1(a_in), cos, sin, key_mask)
        return h + self.mlp(self.norm2(h))


class Proposer(nn.Module):
    """g_psi. Input: frozen decoder embeddings of the BARE question, left-padded ([pad.., x_1..x_n]).
    Output: L0 in R^{K_train x d_dec} (fp32)."""

    def __init__(self, model_cfg: dict, d_dec: int, shell: float):
        super().__init__()
        m, p = model_cfg, model_cfg["proposer"]
        dp, k = int(m["d_prime"]), int(m["K_train"])
        self.d_prime, self.k = dp, k
        self.scale = math.sqrt(dp)
        self.shell_rescale = bool(p["shell_rescale"])
        self.register_buffer("shell", torch.tensor(float(shell)))
        self.down = nn.Linear(d_dec, dp)                                             # P_down
        self.queries = nn.Parameter(trunc_normal_(torch.empty(k, dp), std=float(p["queries_std"])))
        self.blocks = nn.ModuleList(Block(dp, int(m["heads"]), int(p["mlp_hidden"]), float(m["norm_eps"]),
                                          m["block_norm"]) for _ in range(int(p["blocks"])))
        self.final_norm = RMSNorm(dp, float(m["norm_eps"])) if (m["block_norm"] == "pre" and m["final_norm"]) else None
        self.up = nn.Linear(dp, d_dec)                                               # P_up
        self.rotary = Rotary(dp // int(m["heads"]), float(m["rope_theta"]))
        # scale-aware init: unit-size entries after P_down * sqrt(d'); initial |L0_k| ~ the embedding shell
        trunc_normal_(self.down.weight, std=1.0 / (shell * self.scale))
        nn.init.zeros_(self.down.bias)
        trunc_normal_(self.up.weight, std=shell / math.sqrt(d_dec * dp))
        nn.init.zeros_(self.up.bias)

    def forward(self, x_emb: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        b = x_emb.shape[0]
        h = self.down(x_emb.float()) * self.scale
        seq = torch.cat([h, self.queries.unsqueeze(0).expand(b, -1, -1).to(h.dtype)], dim=1)
        mask = torch.cat([x_mask.bool(), torch.ones(b, self.k, dtype=torch.bool, device=x_mask.device)], dim=1)
        pos = (mask.long().cumsum(dim=1) - 1).clamp_min(0)                            # per-example; pads masked
        cos, sin = self.rotary(pos)
        for blk in self.blocks:
            seq = blk(seq, cos, sin, key_mask=mask)
        q = seq[:, -self.k:]
        if self.final_norm is not None:
            q = self.final_norm(q)
        l0 = self.up(q).float()
        if self.shell_rescale:                  # optional: rescale each row to the decoder's mean embedding norm
            l0 = l0 * (self.shell / l0.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return l0


class Refiner(nn.Module):
    """r_phi: TRM recursion over the latents; the question is seen only through L0.

    Initial states z_L^0, z_H^0 are input-independent: `init_state` learned (parameters) or fixed
    (buffers); `init_state_shape` per_slot ([K_train, d'], the first K rows are used when fewer
    latents are refined) or shared ([d'], broadcast over slots). Under the truncated gradient they
    enter only the stop-gradient cycles, so they receive gradient only when S x H = 1."""

    def __init__(self, model_cfg: dict, d_dec: int, shell: float):
        super().__init__()
        m, r = model_cfg, model_cfg["refiner"]
        dp = int(m["d_prime"])
        self.d_prime, self.scale = dp, math.sqrt(dp)
        self.S, self.H, self.T = int(r["S"]), int(r["H"]), int(r["T"])
        self.down = nn.Linear(d_dec, dp)                                             # P'_down
        self.f = Block(dp, int(m["heads"]), int(r["mlp_hidden"]), float(m["norm_eps"]), m["block_norm"])
        self.final_norm = RMSNorm(dp, float(m["norm_eps"])) if (m["block_norm"] == "pre" and m["final_norm"]) else None
        self.up = nn.Linear(dp, d_dec)                                               # P'_up
        self.rotary = Rotary(dp // int(m["heads"]), float(m["rope_theta"]))
        shape = (int(m["K_train"]), dp) if r["init_state_shape"] == "per_slot" else (dp,)
        std = float(r["init_std"])
        zl, zh = trunc_normal_(torch.empty(shape), std=std), trunc_normal_(torch.empty(shape), std=std)
        if r["init_state"] == "learned":
            self.zL0, self.zH0 = nn.Parameter(zl), nn.Parameter(zh)
        else:
            self.register_buffer("zL0", zl)
            self.register_buffer("zH0", zh)
        trunc_normal_(self.down.weight, std=1.0 / (shell * self.scale))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)                                               # Delta = 0 at the start
        nn.init.zeros_(self.up.bias)

    def _state(self, z0: torch.Tensor, b: int, k: int) -> torch.Tensor:
        if z0.dim() == 1:
            return z0.expand(b, k, -1)
        assert k <= z0.shape[0], f"{k} latents > {z0.shape[0]} per-slot initial states"
        return z0[:k].unsqueeze(0).expand(b, -1, -1)

    def _cycle(self, zl, zh, u, cos, sin):
        for _ in range(self.T):
            zl = self.f(zl, cos, sin, cond=zh + u)                                  # z_L <- f(z_L, z_H + u)
        zh = self.f(zh, cos, sin, cond=zl)                                          # z_H <- f(z_H, z_L)
        return zl, zh

    def forward(self, l0: torch.Tensor, S: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """l0: [B, K, d_dec] -> (L*, Delta), both fp32. S overrides the number of outer iterations
        (inference only; the weights do not depend on it)."""
        b, k, _ = l0.shape
        l0 = l0.float()
        u = self.down(l0) * self.scale
        pos = torch.arange(k, device=l0.device).unsqueeze(0).expand(b, -1)
        cos, sin = self.rotary(pos)
        zl, zh = self._state(self.zL0, b, k), self._state(self.zH0, b, k)
        zl, zh = zl.to(u.dtype), zh.to(u.dtype)
        with torch.no_grad():                                   # outputs of these cycles carry no graph
            for _ in range((S or self.S) * self.H - 1):
                zl, zh = self._cycle(zl, zh, u, cos, sin)
        zl, zh = self._cycle(zl, zh, u, cos, sin)                                   # the only differentiated cycle
        top = zh if self.final_norm is None else self.final_norm(zh)
        delta = self.up(top).float()
        return l0 + delta, delta


def proposer_signature(model_cfg: dict) -> dict:
    """The settings that determine the proposer's computation (a stage-1 checkpoint pairs with a stage-2
    config only if these match; initialization scales are irrelevant once trained)."""
    keys = ("d_prime", "heads", "rope_theta", "norm_eps", "block_norm", "final_norm", "K_train")
    sig = {k: model_cfg[k] for k in keys}
    sig["proposer"] = {k: v for k, v in model_cfg["proposer"].items() if k != "queries_std"}
    return sig


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def param_breakdown(proposer: Proposer, refiner: Refiner) -> List[Tuple[str, int]]:
    """Trainable parameters per component."""
    n = lambda *mods: sum(p.numel() for m in mods if m is not None for p in m.parameters())
    rows = [("proposer: input projection P_down", n(proposer.down)),
            ("proposer: query embeddings", proposer.queries.numel()),
            ("proposer: encoder blocks", n(proposer.blocks, proposer.final_norm)),
            ("proposer: output projection P_up", n(proposer.up)),
            ("refiner: input projection P'_down", n(refiner.down)),
            ("refiner: transition block f", n(refiner.f, refiner.final_norm)),
            ("refiner: output projection P'_up", n(refiner.up)),
            ("refiner: initial states z_L^0, z_H^0",
             sum(p.numel() for p in (refiner.zL0, refiner.zH0) if isinstance(p, nn.Parameter)))]
    return rows


def format_breakdown(rows: List[Tuple[str, int]]) -> str:
    total = sum(v for _, v in rows)
    lines = [f"  {name:<40} {v / 1e6:8.3f}M" for name, v in rows]
    return "\n".join(lines + [f"  {'total trainable':<40} {total / 1e6:8.3f}M"])


def state_sha(state: Dict[str, torch.Tensor]) -> str:
    """Content hash of a state dict (names + raw bytes), to pair checkpoints exactly."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
