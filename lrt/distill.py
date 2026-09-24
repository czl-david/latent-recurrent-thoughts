"""Self-distillation: a data-preparation tool, never part of training.

For every training question of the config's task (raw rows, minus duplicates of evaluation questions):
  1. sample the FROZEN decoder on [I; x] (no latents, non-thinking chat template) n times;
  2. keep the samples the task's own scorer marks correct; pick one per question (seeded) and turn it into
     a training target (task.target_from_response);
  3. optionally (--hint), for questions without a correct sample, sample again with the task's gold hint
     appended (task.gold_hint) and keep correct responses that do not betray it (task.hint_leaked);
  4. write jsonl rows {uid, x, target, source, n_correct, n_samples} to $LRT_OUT_ROOT/data_gen/<out>.
Pipeline steps refer to the file as 'gen:<out>'.

    python -m lrt.distill --config CONFIG --out FILE [--n 8] [--hint] [--limit N] [--backend sglang|hf]
"""

import argparse
import json
import os
import random
import time
from typing import Callable, List

from lrt import config as C
from lrt.data import get_task
from lrt.paths import decoder_path, generated_data_dir

SampleFn = Callable[[List[str], int], List[List[str]]]


def log(msg: str) -> None:
    print(f"[distill {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def chat_prompt(tok, user_text: str) -> str:
    text = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    assert text.endswith("<think>\n\n</think>\n\n"), "non-thinking opening expected"
    return text


def distill(cfg, sample_fn: SampleFn, tok, n: int, use_hint: bool, seed: int = 0, limit: int = 0) -> List[dict]:
    task = get_task(cfg)
    rows = task.dedup_against_eval(task.load("train"))
    if limit > 0:
        rows = rows[:limit]
    rng = random.Random(seed)
    log(f"{cfg.task}: {len(rows)} training questions, {n} samples each")
    samples = sample_fn([chat_prompt(tok, task.pre_text(e)) for e in rows], n)
    out, misses = [], []
    for e, outs in zip(rows, samples):
        good = [s for s in outs if task.score(s, e)["correct"]]
        if good:
            out.append({"uid": e.uid, "x": e.x, "target": task.target_from_response(rng.choice(good), e),
                        "source": "sample", "n_correct": len(good), "n_samples": len(outs)})
        else:
            misses.append(e)
    log(f"sampled: {len(out)}/{len(rows)} questions have a correct sample")
    if use_hint and misses:
        hinted = [e for e in misses if task.gold_hint(e) is not None]
        prompts = [chat_prompt(tok, f"{task.pre_text(e)}\n\n{task.gold_hint(e)}") for e in hinted]
        n_ok = 0
        for e, outs in zip(hinted, sample_fn(prompts, n)):
            good = [s for s in outs if task.score(s, e)["correct"] and not task.hint_leaked(s)]
            if good:
                n_ok += 1
                out.append({"uid": e.uid, "x": e.x, "target": task.target_from_response(rng.choice(good), e),
                            "source": "hinted", "n_correct": len(good), "n_samples": len(outs)})
        log(f"hinted: {n_ok}/{len(hinted)} misses answered without leaking the hint")
    return out


def hf_sampler(model, tok, temperature: float, top_p: float, top_k: int, max_new: int, batch: int) -> SampleFn:
    """In-process sampling with the frozen model (no server needed)."""
    import torch
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad = tok.pad_token_id if tok.pad_token_id is not None else stop[0]

    @torch.no_grad()
    def sample(prompts: List[str], n: int) -> List[List[str]]:
        out: List[List[str]] = []
        for i in range(0, len(prompts), batch):
            ids = [tok(p, add_special_tokens=False).input_ids for p in prompts[i:i + batch]]
            width = max(len(x) for x in ids)
            dev = model.device
            input_ids = torch.full((len(ids), width), pad, dtype=torch.long, device=dev)
            mask = torch.zeros((len(ids), width), dtype=torch.long, device=dev)
            for j, x in enumerate(ids):
                input_ids[j, width - len(x):] = torch.tensor(x, device=dev)
                mask[j, width - len(x):] = 1
            gen = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=True, temperature=temperature,
                                 top_p=top_p, top_k=top_k, num_return_sequences=n, max_new_tokens=max_new,
                                 eos_token_id=stop, pad_token_id=pad)[:, width:]
            texts = []
            for row in gen.tolist():
                cut = next((k for k, t in enumerate(row) if t in stop), len(row))
                texts.append(tok.decode(row[:cut], skip_special_tokens=False))
            out.extend(texts[j * n:(j + 1) * n] for j in range(len(ids)))
            log(f"  sampled {min(i + batch, len(prompts))}/{len(prompts)} prompts")
        return out

    return sample


def main() -> None:
    ap = argparse.ArgumentParser(prog="lrt.distill")
    ap.add_argument("--config", required=True, help="an experiment config (its task is distilled)")
    ap.add_argument("--out", required=True, help="file name under $LRT_OUT_ROOT/data_gen/")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--hint", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default=os.environ.get("LRT_EVAL_BACKEND", "sglang"), choices=["sglang", "hf"])
    ap.add_argument("--batch", type=int, default=32, help="prompts per generate call (hf backend)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SGLANG_PORT", "30000")))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(decoder_path())
    cfg = C.load(a.config)
    if a.backend == "hf":
        import torch
        from lrt.decoder import load_frozen_model
        model = load_frozen_model(decoder_path(), torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).eval()
        sample_fn = hf_sampler(model, tok, a.temperature, a.top_p, a.top_k, a.max_new, a.batch)
    else:
        from lrt import sglang_client
        sglang_client.check_health(a.port)

        def sample_fn(prompts: List[str], n: int) -> List[List[str]]:
            return sglang_client.sample_texts(decoder_path(), prompts, a.port, n, a.temperature, a.top_p, a.top_k,
                                              a.max_new)

    rows = distill(cfg, sample_fn, tok, a.n, a.hint, seed=a.seed, limit=a.limit)
    path = os.path.join(generated_data_dir(), a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {len(rows)} targets -> {path} (pipeline steps refer to it as 'gen:{a.out}')")


if __name__ == "__main__":
    main()
