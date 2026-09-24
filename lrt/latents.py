"""The LRT forward pass up to the decoder input:

    x -> E[x] (frozen) -> proposer -> L0 [K_train]
      -> (deploy condition: keep the first K_infer) -> refiner -> L* = L0 + Delta
"""

import contextlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from lrt.modules import Proposer, Refiner


def module_autocast(device: torch.device, precision: str):
    """Compute dtype of the trainable modules (the decoder always runs in bfloat16)."""
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def embed_questions(decoder, xs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bare question texts -> LEFT-padded frozen embeddings [B, n, d] + mask [B, n]."""
    ids = [decoder.tok(x, add_special_tokens=False).input_ids for x in xs]
    n = max(len(i) for i in ids)
    device = decoder.device
    padded = torch.full((len(ids), n), decoder.pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(ids), n), dtype=torch.bool, device=device)
    for b, row in enumerate(ids):
        padded[b, n - len(row):] = torch.tensor(row, device=device)
        mask[b, n - len(row):] = True
    with torch.no_grad():
        emb = decoder.embed(padded)
    return emb, mask


def token_lengths(tok, texts: Sequence[str], chunk: int = 20000) -> List[int]:
    out: List[int] = []
    for i in range(0, len(texts), chunk):
        out.extend(len(ids) for ids in tok(list(texts[i:i + chunk]), add_special_tokens=False)["input_ids"])
    return out


def compute_latents(proposer: Proposer, refiner: Optional[Refiner], x_emb: torch.Tensor, x_mask: torch.Tensor,
                    condition: str, k_infer: int, S: Optional[int] = None) -> Dict[str, Optional[torch.Tensor]]:
    """condition 'train': all K_train latents; 'deploy': the first K_infer (the inference path).
    With a refiner, latents = L* = L0 + Delta."""
    l0 = proposer(x_emb, x_mask)
    if condition == "deploy":
        l0 = l0[:, :k_infer]
    else:
        assert condition == "train", condition
    if refiner is None:
        return {"L0": l0, "latents": l0, "delta": None}
    l_star, delta = refiner(l0, S=S)
    return {"L0": l0, "latents": l_star, "delta": delta}


@torch.no_grad()
def mean_pairwise_cos(z: torch.Tensor) -> float:
    """Mean over latent slots of the mean cosine between DIFFERENT instances ([B, K, d]);
    ~1.0 means the latents are the same for every instance."""
    b = z.shape[0]
    if b < 2:
        return float("nan")
    zn = F.normalize(z.float(), dim=-1)
    sims = torch.einsum("bkd,ckd->kbc", zn, zn)
    off = sims.sum(dim=(1, 2)) - sims.diagonal(dim1=1, dim2=2).sum(-1)
    return float((off / (b * (b - 1))).mean())
