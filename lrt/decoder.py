"""The frozen decoder M: input assembly [I; x; L], answer cross-entropy (full-sequence and
prefix-KV paths) and in-process greedy generation.

Layout of one example:

    <|im_start|>user\n{I}\n\n{x_display}{sep}[L_1..L_K]<|im_end|>\n
    <|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>

Labels are -100 everywhere except the answer tokens and the closing <|im_end|>. M's weights never
receive gradients; gradients flow through its activations into the latents only.
"""

import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer

NONTHINK_OPENING = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def log(msg: str) -> None:
    print(f"[decoder {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nonthink_opening(tok) -> str:
    rendered = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    assert rendered.endswith(NONTHINK_OPENING), (
        f"tokenizer's enable_thinking=False prompt no longer ends with {NONTHINK_OPENING!r}: {rendered!r}")
    return NONTHINK_OPENING


class _ReadOnlyLayer(DynamicLayer):
    """A cache layer holding a fixed prefix. `update` returns prefix + new
    states WITHOUT storing them, so the cache is reusable and a recomputed
    forward (gradient checkpointing) is idempotent."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        super().__init__()
        self.keys, self.values = keys, values
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        return torch.cat([self.keys, key_states], dim=-2), torch.cat([self.values, value_states], dim=-2)


def _read_only_cache(filled: DynamicCache) -> DynamicCache:
    cache = DynamicCache()
    cache.layers = [_ReadOnlyLayer(layer.keys, layer.values) for layer in filled.layers]
    cache.layer_class_to_replicate = None
    return cache


def load_frozen_model(path: str, device: torch.device, grad_ckpt: bool = False, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    log(f"loading frozen decoder {path} ({dtype}, sdpa, grad_ckpt={grad_ckpt}) ...")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    model.to(device)
    return model


class FrozenDecoder(nn.Module):
    """Wraps a (frozen) causal LM + its tokenizer. `model` may be any HF causal
    LM with the Qwen3 chat template (tests use a tiny random Qwen3)."""

    def __init__(self, model: nn.Module, tok, latent_sep: str = " ", grad_ckpt: bool = False):
        super().__init__()
        self.model = model
        self.tok = tok
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.grad_ckpt = grad_ckpt
        if grad_ckpt:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
            self.model.train()      # checkpointing is active only in train mode; Qwen3 has no dropout
        else:
            self.model.eval()
        assert tok.eos_token == "<|im_end|>", f"unexpected eos token {tok.eos_token!r}"
        self.opening = nonthink_opening(tok)
        self.latent_sep = latent_sep
        self.eos_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.eos_id
        self.post_ids = tok("<|im_end|>\n" + self.opening, add_special_tokens=False).input_ids
        with torch.no_grad():
            w = self.embed_weight()
            self.shell = float(w.float().norm(dim=-1).mean())               # mean embedding-row norm
        self.hidden = int(self.embed_weight().shape[1])

    # -- basic pieces -----------------------------------------------------------

    def embed_weight(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    @property
    def device(self) -> torch.device:
        return self.embed_weight().device

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.embed_weight())

    def pre_ids(self, pre_text: str) -> List[int]:
        """Token ids of the user turn up to the latents: '<|im_start|>user\\n' + pre_text + sep."""
        return self.tok(f"<|im_start|>user\n{pre_text}{self.latent_sep}", add_special_tokens=False).input_ids

    def answer_ids(self, answer: str) -> List[int]:
        return self.tok(answer + self.tok.eos_token, add_special_tokens=False).input_ids

    # -- teacher-forced cross-entropy ---------------------------------------------

    def ce(self, pre_texts: Sequence[str], latents: Optional[torch.Tensor], answers: Sequence[str],
           use_prefix_cache: bool = True, max_answer_tokens: int = 0) -> Dict[str, torch.Tensor]:
        """Answer CE given [pre; latents; post; answer]. latents: [B, K, d] (any
        float dtype, cast to the decoder dtype) or None (no latents).

        Returns dict(loss = token-mean CE, token_acc, answer_em (per-batch mean),
        em_per_example [B] bool, n_tokens, nll_per_example [B])."""
        b = len(pre_texts)
        pre = [self.pre_ids(t) for t in pre_texts]
        ans = [self.answer_ids(a) for a in answers]
        if max_answer_tokens:
            longest = max(len(a) for a in ans)
            assert longest <= max_answer_tokens, f"answer of {longest} tokens > max_answer_tokens={max_answer_tokens}"
        if latents is not None:
            assert latents.shape[0] == b and latents.shape[2] == self.hidden
        if use_prefix_cache and self.grad_ckpt:
            # HF drops past_key_values inside checkpointed layers (and refuses to fill a
            # cache in train mode): the suffix would silently lose the prefix.
            raise ValueError("prefix-KV caching and decoder gradient checkpointing are mutually exclusive; "
                             "set train.prefix_cache=false or train.grad_ckpt=false")
        if use_prefix_cache:
            hidden, labels = self._forward_prefix_cached(pre, latents, ans)
        else:
            hidden, labels = self._forward_full(pre, latents, ans)
        return self._ce_from_hidden(hidden, labels)

    def _suffix_embeds(self, latents_b: Optional[torch.Tensor], ans_ids: List[int]):
        device, dtype = self.device, self.embed_weight().dtype
        post = self.embed(torch.tensor(self.post_ids, device=device))
        a = torch.tensor(ans_ids, device=device)
        parts = ([latents_b.to(dtype)] if latents_b is not None else []) + [post, self.embed(a)]
        emb = torch.cat(parts, dim=0)
        n_lat = 0 if latents_b is None else latents_b.shape[0]
        labels = torch.full((emb.shape[0],), -100, dtype=torch.long, device=device)
        labels[n_lat + len(self.post_ids):] = a
        return emb, labels

    def _forward_full(self, pre: List[List[int]], latents, ans: List[List[int]]):
        """Whole sequence in one pass, right-padded (the path used with gradient checkpointing)."""
        device, dtype = self.device, self.embed_weight().dtype
        seqs, labs = [], []
        for i in range(len(pre)):
            p = torch.tensor(pre[i], device=device)
            s_emb, s_lab = self._suffix_embeds(None if latents is None else latents[i], ans[i])
            seqs.append(torch.cat([self.embed(p), s_emb], dim=0))
            labs.append(torch.cat([torch.full((len(pre[i]),), -100, dtype=torch.long, device=device), s_lab]))
        emb, mask, labels = self._right_pad(seqs, labs, dtype)
        out = self.model.model(inputs_embeds=emb, attention_mask=mask, use_cache=False)
        return out.last_hidden_state, labels

    def _forward_prefix_cached(self, pre: List[List[int]], latents, ans: List[List[int]]):
        """The prefix [I; x] depends on no trainable parameter: compute its KV
        under no_grad (right-padded, so no attention row is fully masked), then
        run forward+backward over [latents; post; answer] only. Suffix positions
        continue each example's true prefix length."""
        device, dtype = self.device, self.embed_weight().dtype
        b = len(pre)
        p_len = torch.tensor([len(p) for p in pre], device=device)
        p_max = int(p_len.max())
        pre_ids = torch.full((b, p_max), self.pad_id, dtype=torch.long, device=device)
        pre_mask = torch.zeros((b, p_max), dtype=torch.long, device=device)
        for i, p in enumerate(pre):
            pre_ids[i, : len(p)] = torch.tensor(p, device=device)
            pre_mask[i, : len(p)] = 1
        with torch.no_grad():
            filled = DynamicCache()
            # inner model only: the prefix needs its KV, not its logits
            self.model.model(input_ids=pre_ids, attention_mask=pre_mask, past_key_values=filled, use_cache=True)
        cache = _read_only_cache(filled)

        seqs, labs = [], []
        for i in range(b):
            s_emb, s_lab = self._suffix_embeds(None if latents is None else latents[i], ans[i])
            seqs.append(s_emb)
            labs.append(s_lab)
        emb, s_mask, labels = self._right_pad(seqs, labs, dtype)
        s_max = emb.shape[1]
        positions = p_len[:, None] + torch.arange(s_max, device=device)[None, :]
        attn = torch.cat([pre_mask, s_mask], dim=1)
        out = self.model.model(inputs_embeds=emb, attention_mask=attn, position_ids=positions,
                               past_key_values=cache, use_cache=False)
        return out.last_hidden_state, labels

    def _right_pad(self, seqs: List[torch.Tensor], labs: List[torch.Tensor], dtype):
        device = self.device
        max_len = max(s.shape[0] for s in seqs)
        pad_emb = self.embed(torch.tensor([self.pad_id], device=device))[0]
        emb = pad_emb.expand(len(seqs), max_len, -1).clone().to(dtype)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        labels = torch.full((len(seqs), max_len), -100, dtype=torch.long, device=device)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            emb[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = 1
            labels[i, : l.shape[0]] = l
        return emb, mask, labels

    def _ce_from_hidden(self, hidden: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Token-mean CE with the LM head applied ONLY at supervised positions
        (full-vocabulary fp32 logits at every position are the main memory cost
        of training through the decoder)."""
        b = hidden.shape[0]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        h_sel = hidden[:, :-1][mask]                                        # [N, d]
        target = shift_labels[mask]                                         # [N]
        logits = self.model.lm_head(h_sel).float()
        tok_nll = F.cross_entropy(logits, target, reduction="none")
        n_tok = mask.sum()
        loss = tok_nll.sum() / n_tok.clamp_min(1)
        with torch.no_grad():
            ex = mask.nonzero()[:, 0]
            per_count = mask.sum(-1)
            per_correct = torch.zeros(b, device=hidden.device).index_add_(0, ex, (logits.argmax(-1) == target).float())
            per_nll = torch.zeros(b, device=hidden.device).index_add_(0, ex, tok_nll.detach())
            em = (per_correct == per_count) & (per_count > 0)
            out = dict(
                token_acc=per_correct.sum() / n_tok.clamp_min(1),
                answer_em=em.float().mean(),
                em_per_example=em,
                n_tokens=n_tok,
                nll_per_example=per_nll / per_count.clamp_min(1),
            )
        out["loss"] = loss
        return out

    # -- inference -----------------------------------------------------------------

    @torch.no_grad()
    def prefix_embeds(self, pre_texts: Sequence[str], latents: Optional[torch.Tensor]) -> List[torch.Tensor]:
        """Per-example [L_b, d] input embeddings up to and including the assistant
        opening (for SGLang `input_embeds` or in-process generation)."""
        device, dtype = self.device, self.embed_weight().dtype
        post = self.embed(torch.tensor(self.post_ids, device=device))
        out = []
        for i, t in enumerate(pre_texts):
            parts = [self.embed(torch.tensor(self.pre_ids(t), device=device))]
            if latents is not None:
                parts.append(latents[i].to(dtype))
            parts.append(post)
            out.append(torch.cat(parts, dim=0))
        return out

    @torch.no_grad()
    def generate_greedy(self, prefixes: List[torch.Tensor], max_new_tokens: int) -> List[str]:
        """In-process greedy decoding from per-example prefix embeddings
        (left-padded batch). Stops at <|im_end|> / <|endoftext|>."""
        device, dtype = self.device, self.embed_weight().dtype
        b = len(prefixes)
        max_len = max(p.shape[0] for p in prefixes)
        pad_emb = self.embed(torch.tensor([self.pad_id], device=device))[0]
        emb = pad_emb.expand(b, max_len, -1).clone().to(dtype)
        mask = torch.zeros((b, max_len), dtype=torch.long, device=device)
        for i, p in enumerate(prefixes):
            emb[i, max_len - p.shape[0]:] = p
            mask[i, max_len - p.shape[0]:] = 1
        stop = [self.eos_id]
        eot = self.tok.convert_tokens_to_ids("<|endoftext|>")
        if isinstance(eot, int) and eot >= 0:
            stop.append(eot)
        was_training = self.model.training
        self.model.eval()
        gen = self.model.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens,
                                  do_sample=False, eos_token_id=stop, pad_token_id=self.pad_id,
                                  temperature=None, top_p=None, top_k=None)
        if was_training:
            self.model.train()
        texts = []
        for row in gen.tolist():
            cut = [t for t in row]
            for j, t in enumerate(cut):
                if t in stop:
                    cut = cut[:j]
                    break
            texts.append(self.tok.decode(cut, skip_special_tokens=False))
        return texts


def build_decoder(path: str, device: torch.device, latent_sep: str, grad_ckpt: bool = False,
                  dtype=torch.bfloat16) -> FrozenDecoder:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = load_frozen_model(path, device, grad_ckpt=grad_ckpt, dtype=dtype)
    dec = FrozenDecoder(model, tok, latent_sep=latent_sep, grad_ckpt=grad_ckpt)
    log(f"decoder ready: hidden={dec.hidden} shell(mean |E_row|)={dec.shell:.4f} opening={dec.opening!r}")
    return dec


class EmbedOnlyDecoder:
    """Evaluation-side stand-in (the SGLang server holds the weights): exposes
    prefix_embeds() with only the embedding table + tokenizer."""

    def __init__(self, embed_weight: torch.Tensor, tok, latent_sep: str = " "):
        self._w = embed_weight
        self.tok = tok
        self.opening = nonthink_opening(tok)
        self.latent_sep = latent_sep
        self.eos_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.eos_id
        self.post_ids = tok("<|im_end|>\n" + self.opening, add_special_tokens=False).input_ids
        self.shell = float(embed_weight.float().norm(dim=-1).mean())
        self.hidden = int(embed_weight.shape[1])
        self.model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=self._w))

    embed_weight = FrozenDecoder.embed_weight
    device = FrozenDecoder.device
    embed = FrozenDecoder.embed
    pre_ids = FrozenDecoder.pre_ids
    prefix_embeds = FrozenDecoder.prefix_embeds
