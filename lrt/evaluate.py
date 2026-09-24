"""Evaluation along the inference path: proposer -> first K_infer latents -> refiner (full S x H recursion)
-> greedy decoding of [I; x; L*] by the frozen decoder -> the task's scorer. Also the decoder-only anchor
([I; x], no latents) and the proposer-only ablation (the stage-1 checkpoint, no refiner).

The modules are rebuilt from the configurations stored in their checkpoints; the current config supplies
the task, model.K_infer and (optionally) a test-time number of refiner outer iterations S.

Writes <run_dir>/eval/<split>_<mode>_k<K_infer>[_S<S>][_last]_<backend>.jsonl and a _summary.json
(anchors: $LRT_OUT_ROOT/<task>/anchors/). The split "holdout" is the held-out training rows that
checkpoint selection uses.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from lrt.data import get_task
from lrt.decoder import EmbedOnlyDecoder, build_decoder
from lrt.latents import compute_latents, embed_questions, module_autocast
from lrt.modules import Proposer, Refiner, state_sha
from lrt.paths import decoder_path, out_root


def log(msg: str) -> None:
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_embed_weight(path: str, device: torch.device) -> torch.Tensor:
    """Only the decoder's input-embedding table (the SGLang server holds the rest)."""
    from safetensors import safe_open
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"]["model.embed_tokens.weight"]
    with safe_open(os.path.join(path, shard), framework="pt", device="cpu") as f:
        return f.get_tensor("model.embed_tokens.weight").to(device)


def load_modules(cfg, d_dec: int, shell: float, device, stage1_only: bool, ckpt: str = "best"):
    """The refiner is paired with EXACTLY the proposer it was trained against (path + content hash)."""
    run_dir = cfg.run_dir()
    ck2 = None
    if stage1_only:
        p1 = os.path.join(run_dir, "stage1", f"{ckpt}.pt")
    else:
        ck2 = torch.load(os.path.join(run_dir, "stage2", f"{ckpt}.pt"), map_location="cpu", weights_only=False)
        p1 = ck2["stage1_ckpt"]
    ck1 = torch.load(p1, map_location="cpu", weights_only=False)
    if ck2 is not None and state_sha(ck1["eval_module"]) != ck2["stage1_sha"]:
        raise RuntimeError(f"{p1} changed after the refiner was trained against it (content hash mismatch)")
    m1 = ck1["config"]["config"]["model"]
    if int(cfg["model"]["K_infer"]) > int(m1["K_train"]):
        raise ValueError(f"model.K_infer={cfg['model']['K_infer']} > the checkpoint's K_train={m1['K_train']}")
    proposer = Proposer(m1, d_dec, shell).to(device).eval()
    proposer.load_state_dict(ck1["eval_module"])
    info = {"stage1": {"path": p1, "step": ck1["step"]}}
    refiner = None
    if ck2 is not None:
        m2 = ck2["config"]["config"]["model"]
        refiner = Refiner(m2, d_dec, shell).to(device).eval()
        refiner.load_state_dict(ck2["eval_module"])
        info["stage2"] = {"path": os.path.join(run_dir, "stage2", f"{ckpt}.pt"), "step": ck2["step"],
                          "trained_S": int(m2["refiner"]["S"])}
    return proposer, refiner, info


def eval_examples(cfg, task, split: str, tok) -> list:
    """An evaluation split, or "holdout": the held-out training rows of checkpoint selection (same length
    caps and train.eval_holdout_n as training)."""
    if split != "holdout":
        return task.load(split)
    from lrt.train import filter_by_length
    _, held, _ = task.train_pool_and_holdout()
    caps = (int(cfg["data"]["max_x_tokens"]), int(cfg["answer"]["max_tokens"]))
    return filter_by_length(held, tok, *caps, what="holdout")[: int(cfg["train"]["eval_holdout_n"])]


@torch.no_grad()
def run_eval(cfg, split: str, backend: str = "sglang", port: int = 30000, anchor: bool = False,
             stage1_only: bool = False, limit: int = 0, infer_S: int = 0, ckpt: str = "best",
             out_dir: Optional[str] = None, max_in_flight: int = 32, decoder=None) -> Dict[str, Any]:
    """decoder: an already-built FrozenDecoder for backend='hf' (tests); otherwise built from
    $LRT_DECODER_PATH (hf) or only its embedding table (sglang)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task = get_task(cfg)
    sep = cfg["chat"]["latent_separator"]
    if decoder is not None:
        tok = decoder.tok
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(decoder_path())
    examples = eval_examples(cfg, task, split, tok)
    if limit > 0:
        examples = examples[:limit]
    if backend == "sglang":
        from lrt import sglang_client
        sglang_client.check_health(port)
        dec = EmbedOnlyDecoder(load_embed_weight(decoder_path(), device), tok, latent_sep=sep)
    else:
        assert backend == "hf", backend
        dec = decoder or build_decoder(decoder_path(), device, sep)
        device = dec.device
    mode = "anchor" if anchor else ("stage1_only" if stage1_only else "lrt")
    proposer = refiner = None
    info: Dict[str, Any] = {}
    if not anchor:
        proposer, refiner, info = load_modules(cfg, dec.hidden, dec.shell, device, stage1_only, ckpt=ckpt)
    k_infer = int(cfg["model"]["K_infer"])
    max_new = int(cfg["eval"]["max_new_tokens"])
    bs = int(cfg["eval"]["batch_size"])
    log(f"{cfg.task}/{split} mode={mode} backend={backend} n={len(examples)} K_infer={k_infer} "
        + (f"infer_S={infer_S} " if infer_S else "") + f"max_new={max_new} | {info}")

    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        latents = None
        if not anchor:
            x_emb, x_mask = embed_questions(dec, [e.x for e in chunk])
            with module_autocast(device, cfg["train"]["precision"]):
                latents = compute_latents(proposer, refiner, x_emb, x_mask, "deploy", k_infer,
                                          S=infer_S or None)["latents"]
        prefixes = dec.prefix_embeds([task.pre_text(e) for e in chunk], latents)
        if backend == "sglang":
            from lrt import sglang_client
            texts = sglang_client.decode_greedy(decoder_path(), prefixes, port, max_new, max_in_flight)
        else:
            texts = dec.generate_greedy(prefixes, max_new)
        for e, t, s in zip(chunk, texts, task.score_many(texts, chunk)):
            records.append({"uid": e.uid, "raw": t, **s})
        log(f"  {len(records)}/{len(examples)} decoded ({time.time() - t0:.0f}s)")

    summary = task.summarize(records)
    summary.update(task=cfg.task, run=None if anchor else cfg.name, split=split, mode=mode, backend=backend,
                   K_infer=None if anchor else k_infer, infer_S=infer_S or None, ckpt=None if anchor else ckpt,
                   checkpoints=info, config=cfg.to_dict(), wall_s=time.time() - t0)
    if out_dir is None:
        out_dir = (os.path.join(out_root(), cfg.task, "anchors") if anchor
                   else os.path.join(cfg.run_dir(), "eval"))
    os.makedirs(out_dir, exist_ok=True)
    tag = "" if anchor else f"_k{k_infer}" + (f"_S{infer_S}" if infer_S else "") + ("" if ckpt == "best" else f"_{ckpt}")
    stem = os.path.join(out_dir, f"{split}_{mode}{tag}_{backend}" + (f"_n{limit}" if limit > 0 else ""))
    with open(stem + ".jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(stem + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"RESULT {cfg.task}/{split} {mode}: {summary['correct']}/{summary['n']} = {summary['accuracy']:.4f} | "
        f"{stem}.jsonl")
    return summary
