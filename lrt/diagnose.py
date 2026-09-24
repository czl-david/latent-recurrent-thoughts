"""Do a run's latents carry INSTANCE-SPECIFIC information that the decoder uses?

Teacher-forced answer loss / token accuracy / exact match on the held-out training rows, with
    own       the run's own latents (deploy or train condition)
    shuffled  the same latents rolled across examples (distribution kept, instance information removed)
    mean      every example gets the batch-mean latent (a pure soft prompt)
    none      no latents at all ([I; x] only)

    python -m lrt.diagnose --config CONFIG [--stage 1|2] [--n 256] [--condition deploy|train] [--ckpt best|last]
                                           [--pipeline-stage N]   (first transform the rows with stage N's pipeline)
"""

import argparse
import random

import torch

from lrt import config as C
from lrt.data import get_task
from lrt.decoder import build_decoder
from lrt.evaluate import load_modules
from lrt.latents import compute_latents, embed_questions, mean_pairwise_cos, module_autocast
from lrt.paths import decoder_path


@torch.no_grad()
def diagnose(cfg, stage: int, n: int, condition: str, bs: int = 16, decoder=None, ckpt: str = "best",
             none_only: bool = False, pipeline_stage: int = 0) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task = get_task(cfg)
    dec = decoder or build_decoder(decoder_path(), device, cfg["chat"]["latent_separator"])
    info = {}
    if not none_only:
        proposer, refiner, info = load_modules(cfg, dec.hidden, dec.shell, dec.device, stage1_only=(stage == 1),
                                               ckpt=ckpt)
    rows = task.train_pool_and_holdout()[1][:n]
    if pipeline_stage:
        pipe = task.pipeline(cfg.pipeline_name(pipeline_stage))
        rng = random.Random(12345)
        rows = [pipe.sample(e, rng, 0.0) for e in rows]
    names = ("none",) if none_only else ("own", "shuffled", "mean", "none")
    tot = {k: [0.0, 0.0, 0.0, 0] for k in names}                      # nll*tok, tok, correct*tok, em
    cos = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        if none_only:
            variants = {"none": None}
        else:
            x_emb, m = embed_questions(dec, [e.x for e in chunk])
            with module_autocast(dec.device, cfg["train"]["precision"]):
                lat = compute_latents(proposer, refiner, x_emb, m, condition, int(cfg["model"]["K_infer"]))["latents"]
            cos.append(mean_pairwise_cos(lat))
            variants = {"own": lat, "shuffled": lat.roll(1, dims=0),
                        "mean": lat.mean(0, keepdim=True).expand_as(lat).contiguous(), "none": None}
        pre, ans = [task.pre_text(e) for e in chunk], [e.answer for e in chunk]
        for k, v in variants.items():
            out = dec.ce(pre, v, ans)
            t = float(out["n_tokens"])
            tot[k][0] += float(out["loss"]) * t
            tot[k][1] += t
            tot[k][2] += float(out["token_acc"]) * t
            tot[k][3] += int(out["em_per_example"].sum())
    res = {k: {"loss": v[0] / max(v[1], 1), "tok_acc": v[2] / max(v[1], 1), "em": v[3] / max(len(rows), 1)}
           for k, v in tot.items()}
    res["cos_latents"] = sum(cos) / len(cos) if cos else float("nan")
    res["checkpoints"] = info
    return res


def main() -> None:
    ap = argparse.ArgumentParser(prog="lrt.diagnose")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--condition", default="deploy", choices=["deploy", "train"])
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--none-only", action="store_true", help="no checkpoint: only the no-latent baseline")
    ap.add_argument("--pipeline-stage", type=int, default=0, choices=[0, 1, 2])
    a = ap.parse_args()
    cfg = C.load(a.config, a.sets)
    res = diagnose(cfg, a.stage, a.n, a.condition, ckpt=a.ckpt, none_only=a.none_only, pipeline_stage=a.pipeline_stage)
    print(f"[diagnose] {cfg.task}/{cfg.name} stage {a.stage} ({a.condition}, n={a.n}) cos(latents)={res['cos_latents']:.4f}")
    for k in ("own", "shuffled", "mean", "none"):
        if k in res:
            r = res[k]
            print(f"  {k:>8}: loss={r['loss']:.4f} tok_acc={r['tok_acc']:.4f} em={r['em']:.4f}")
    if "own" in res:
        print(f"  instance-information gap (shuffled - own loss) = {res['shuffled']['loss'] - res['own']['loss']:+.4f}")


if __name__ == "__main__":
    main()
