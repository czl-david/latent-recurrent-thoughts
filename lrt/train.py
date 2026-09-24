"""Two-stage training, single GPU or DDP (torchrun).

    stage 1: train the proposer alone; decoder input [I; x; L0], L0 = all K_train latents.
    stage 2: freeze the proposer (its stage-1 checkpoint) and train the refiner on
             CE([I; x; L*]) + lambda * ||Delta||^2 with all K_train latents. L0 is recomputed under
             no_grad for every batch, which equals caching it but also works with per-example pipelines.

Only the decoder's answer cross-entropy trains the modules. stage<N>.aux_readout (off in base.yaml) adds an
optional auxiliary loss that is not part of the method; see `_aux_readout`.

Held-out TRAIN rows are evaluated under two conditions: 'train' (K_train latents) and 'deploy' (the
inference path: the first K_infer latents, then the refiner in stage 2). Checkpoints (best.pt by the
configured selection metric, last.pt with optimizer state) go to <run_dir>/stage<N>/.
"""

import datetime
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from lrt.data import get_task
from lrt.decoder import build_decoder
from lrt.latents import compute_latents, embed_questions, mean_pairwise_cos, module_autocast, token_lengths
from lrt.modules import (Proposer, Refiner, count_params, format_breakdown, param_breakdown, proposer_signature,
                         state_sha)
from lrt.paths import decoder_path


class Dist:
    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world > 1 and not dist.is_initialized():
            # rank 0 evaluates (minutes) while the others wait at a barrier
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                    timeout=datetime.timedelta(hours=2))
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.main = self.rank == 0

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def all_mean(self, value: float) -> float:
        if self.world == 1:
            return value
        t = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(t)
        return float(t.item()) / self.world

    def close(self):
        if self.world > 1 and dist.is_initialized():
            dist.destroy_process_group()


_DIST: Optional[Dist] = None


def log(msg: str) -> None:
    if _DIST is None or _DIST.main:
        print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warmup_steps(sc: Dict[str, Any], total: int) -> int:
    return int(sc["warmup_steps"]) if int(sc["warmup_steps"]) > 0 else int(round(float(sc["warmup_frac"]) * total))


def lr_at(step: int, total: int, sc: Dict[str, Any]) -> float:
    base, warm = float(sc["lr"]), warmup_steps(sc, total)
    if step <= warm:
        return base * step / max(1, warm)
    if sc["schedule"] == "constant":
        return base
    progress = (step - warm) / max(1, total - warm)
    floor = float(sc["min_lr_ratio"])
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


class EMA:
    def __init__(self, module: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        for n, p in module.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def swap(self, module: torch.nn.Module):
        """Swap EMA and live weights in place (call twice to restore)."""
        with torch.no_grad():
            for n, p in module.named_parameters():
                if n in self.shadow:
                    tmp = p.detach().clone()
                    p.copy_(self.shadow[n])
                    self.shadow[n].copy_(tmp)


def filter_by_length(examples, tok, max_x: int, max_answer: int, what: str):
    """Drop rows whose question or target exceeds the configured caps (logged)."""
    x_len = token_lengths(tok, [e.x for e in examples])
    a_len = token_lengths(tok, [e.answer + tok.eos_token for e in examples])
    kept = [e for e, xl, al in zip(examples, x_len, a_len) if xl <= max_x and al <= max_answer]
    if len(kept) < len(examples):
        log(f"{what}: dropped {len(examples) - len(kept)}/{len(examples)} rows over the length caps "
            f"(x <= {max_x} tokens, answer <= {max_answer} tokens)")
    return kept


class BatchStream:
    """Deterministic, rank-consistent sampling: every rank walks the same seeded per-epoch permutations
    and takes its own slice of each global batch."""

    def __init__(self, n: int, per_rank: int, world: int, rank: int, seed: int):
        self.n, self.per_rank, self.world, self.rank, self.seed = n, per_rank, world, rank, seed
        self.global_batch = per_rank * world
        assert self.global_batch <= n, f"global batch {self.global_batch} > pool size {n}"

    def indices(self, global_batch_idx: int) -> List[int]:
        per_epoch = self.n // self.global_batch
        epoch, slot = divmod(global_batch_idx, per_epoch)
        perm = np.random.default_rng(self.seed * 1_000_003 + epoch).permutation(self.n)
        start = slot * self.global_batch + self.rank * self.per_rank
        return perm[start:start + self.per_rank].tolist()


def stage1_checkpoint_path(cfg) -> str:
    sc = cfg.stage(2)
    return os.path.join(cfg.run_dir(sc["stage1_from"]), "stage1", f"{sc['stage1_ckpt']}.pt")


class StageTrainer:
    def __init__(self, cfg, stage: int, decoder=None, stage1_ckpt: Optional[str] = None):
        global _DIST
        assert stage in (1, 2)
        self.cfg, self.stage = cfg, stage
        self.ctx = _DIST = Dist()
        self.tc, self.sc, self.m = cfg["train"], cfg.stage(stage), cfg["model"]
        self.run_dir = cfg.run_dir()
        self.stage_dir = os.path.join(self.run_dir, f"stage{stage}")
        os.makedirs(self.stage_dir, exist_ok=True)
        torch.manual_seed(int(self.tc["seed"]))

        self.task = get_task(cfg)
        self.pipe = self.task.pipeline(cfg.pipeline_name(stage))
        self.decoder = decoder or build_decoder(decoder_path(), self.ctx.device, cfg["chat"]["latent_separator"],
                                                grad_ckpt=bool(self.tc["grad_ckpt"]))
        d_dec, shell = self.decoder.hidden, self.decoder.shell
        self.refiner: Optional[Refiner] = None
        if stage == 1:
            self.proposer = Proposer(self.m, d_dec, shell).to(self.ctx.device)
            self.trainable = self.proposer
        else:
            path = os.path.abspath(stage1_ckpt or stage1_checkpoint_path(cfg))
            ck = torch.load(path, map_location="cpu", weights_only=False)
            saved_model = ck["config"]["config"]["model"]
            if proposer_signature(saved_model) != proposer_signature(self.m):
                raise ValueError(f"{path}: the proposer settings under model: differ from this config's "
                                 f"(checkpoint {proposer_signature(saved_model)} vs {proposer_signature(self.m)})")
            self.proposer = Proposer(saved_model, d_dec, shell).to(self.ctx.device)
            self.proposer.load_state_dict(ck["eval_module"])
            self.proposer.eval()
            for p in self.proposer.parameters():
                p.requires_grad_(False)
            self.stage1_ckpt, self.stage1_sha = path, state_sha(ck["eval_module"])
            log(f"stage 2: proposer frozen from {path} (step {ck['step']})")
            self.refiner = Refiner(self.m, d_dec, shell).to(self.ctx.device)
            r = self.m["refiner"]
            if isinstance(self.refiner.zL0, torch.nn.Parameter) and int(r["S"]) * int(r["H"]) > 1:
                # learned initial states enter only the stop-gradient cycles, so no gradient can reach them;
                # freezing them changes nothing numerically and keeps DDP from waiting for their gradients
                self.refiner.zL0.requires_grad_(False)
                self.refiner.zH0.requires_grad_(False)
                log("stage 2: the learned initial refiner states receive no gradient when S x H > 1; frozen")
            self.trainable = self.refiner
        self.model = (DDP(self.trainable, device_ids=[self.ctx.local_rank] if torch.cuda.is_available() else None)
                      if self.ctx.world > 1 else self.trainable)
        self.opt = torch.optim.AdamW([p for p in self.trainable.parameters() if p.requires_grad],
                                     lr=float(self.sc["lr"]), betas=tuple(float(b) for b in self.sc["betas"]),
                                     eps=float(self.sc["eps"]), weight_decay=float(self.sc["weight_decay"]))
        self.ema = EMA(self.trainable, float(self.sc["ema_decay"])) if float(self.sc["ema_decay"]) > 0 else None

        tok = self.decoder.tok
        pool, holdout, self.is_probe = self.task.train_pool_and_holdout()
        caps = (int(cfg["data"]["max_x_tokens"]), int(cfg["answer"]["max_tokens"]))
        self.pool = filter_by_length(self.pipe.prepare(pool), tok, *caps, what="train pool")
        self.holdout = filter_by_length(holdout, tok, *caps, what="holdout")[: int(self.tc["eval_holdout_n"])]
        self.batch_size = cfg.batch_size(stage)
        self.stream = BatchStream(len(self.pool), self.batch_size, self.ctx.world, self.ctx.rank, int(self.tc["seed"]))
        accum = int(self.tc["grad_accum"])
        if int(self.sc["steps"]) > 0:
            self.total_steps = int(self.sc["steps"])
        else:
            self.total_steps = max(1, math.ceil(float(self.sc["epochs"]) * len(self.pool)
                                                / (self.stream.global_batch * accum)))
        self.metrics_path = os.path.join(self.stage_dir, "metrics.jsonl")

    def autocast(self):
        return module_autocast(self.ctx.device, self.tc["precision"])

    # -- forward --------------------------------------------------------------------------

    def _latents(self, batch, condition: str, train: bool) -> Dict[str, Any]:
        x_emb, x_mask = embed_questions(self.decoder, [e.x for e in batch])
        k_infer = int(self.m["K_infer"])
        with self.autocast():
            if self.stage == 1:
                return compute_latents(self.model if train else self.proposer, None, x_emb, x_mask, condition, k_infer)
            with torch.no_grad():
                l0 = self.proposer(x_emb, x_mask)
            if condition == "deploy":
                l0 = l0[:, :k_infer]
            l_star, delta = (self.model if train else self.refiner)(l0)
        return {"L0": l0, "latents": l_star, "delta": delta}

    def _forward(self, batch, condition: str, train: bool, step: Optional[int] = None) -> Dict[str, Any]:
        lat = self._latents(batch, condition, train)
        ac = self.sc["aux_readout"]
        aux_on = float(ac["weight"]) > 0
        # compute knobs of the auxiliary readout (training only): the decoder CE on the first ce_batch rows,
        # every ce_every steps; the readout covers the whole batch every step
        ce_rows = len(batch)
        if train and aux_on and int(ac["ce_batch"]) > 0:
            ce_rows = min(ce_rows, int(ac["ce_batch"]))
        do_ce = not (train and aux_on and step is not None and step % int(ac["ce_every"]) != 0)
        out = None
        loss = lat["latents"].new_zeros(())
        if do_ce:
            sub = batch[:ce_rows]
            out = self.decoder.ce([self.task.pre_text(e) for e in sub], lat["latents"][:ce_rows],
                                  [e.answer for e in sub], use_prefix_cache=bool(self.tc["prefix_cache"]))
            loss = out["loss"]
        penalty = None
        if self.stage == 2:
            penalty = lat["delta"].float().pow(2).sum(dim=(1, 2)).mean()          # ||Delta||_F^2 per example
            loss = loss + float(self.m["lambda_delta"]) * penalty
        aux = None
        if aux_on:
            aux = self._aux_readout(lat["latents"], batch, float(ac["scale"]))
            loss = loss + float(ac["weight"]) * aux["loss"]
        return {"loss": loss, "ce": out, "lat": lat, "penalty": penalty, "aux": aux}

    def _aux_readout(self, latents: torch.Tensor, batch, scale: float) -> Dict[str, torch.Tensor]:
        """Optional auxiliary loss (stage<N>.aux_readout.weight > 0), not part of the method: latent k is read
        out as answer token k through the FROZEN decoder's input-embedding table,
        logits = scale * cos(latent_k, E_v) over the whole vocabulary, for k < min(K, answer tokens).
        It adds no parameters and no task-specific output layer."""
        emb = self.decoder.embed_weight()
        ids = [self.decoder.answer_ids(e.answer) for e in batch]
        k = latents.shape[1]
        rows, cols, tgt = [], [], []
        for b, a in enumerate(ids):
            n = min(k, len(a))
            rows += [b] * n
            cols += list(range(n))
            tgt += a[:n]
        sel = F.normalize(latents[rows, cols].float(), dim=-1)
        e_n = F.normalize(emb.float(), dim=-1)
        logits = scale * (sel.to(emb.dtype) @ e_n.to(emb.dtype).T).float()
        target = torch.tensor(tgt, device=latents.device)
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return {"loss": loss, "acc": acc}

    # -- evaluation -----------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        mod = self.trainable
        if self.ema is not None:
            self.ema.swap(mod)
        was_training = mod.training
        mod.eval()
        bs = self.batch_size
        aux_on = float(self.sc["aux_readout"]["weight"]) > 0
        res: Dict[str, Any] = {}
        for condition in ("train", "deploy"):
            tot_nll = tot_tok = tot_correct = shuf_nll = aux_acc = 0.0
            em = 0
            for i in range(0, len(self.holdout), bs):
                chunk = self.holdout[i:i + bs]
                f = self._forward(chunk, condition, train=False)
                if f["aux"] is not None:
                    aux_acc += float(f["aux"]["acc"]) * len(chunk)
                n_tok = float(f["ce"]["n_tokens"])
                tot_nll += float(f["ce"]["loss"]) * n_tok
                tot_tok += n_tok
                tot_correct += float(f["ce"]["token_acc"]) * n_tok
                em += int(f["ce"]["em_per_example"].sum())
                if len(chunk) > 1:
                    # instance-information probe: the same latents, rolled across examples
                    shuf = self.decoder.ce([self.task.pre_text(e) for e in chunk], f["lat"]["latents"].roll(1, dims=0),
                                           [e.answer for e in chunk], use_prefix_cache=bool(self.tc["prefix_cache"]))
                    shuf_nll += float(shuf["loss"]) * float(shuf["n_tokens"])
                if i == 0:
                    res[f"cos_L0_{condition}"] = mean_pairwise_cos(f["lat"]["L0"])
                    res[f"norm_L0_{condition}"] = float(f["lat"]["L0"].norm(dim=-1).mean())
                    if f["lat"]["delta"] is not None:
                        res[f"cos_delta_{condition}"] = mean_pairwise_cos(f["lat"]["delta"])
                        res[f"norm_delta_{condition}"] = float(f["lat"]["delta"].norm(dim=-1).mean())
            n = max(len(self.holdout), 1)
            res[f"tf_loss_{condition}"] = tot_nll / max(tot_tok, 1)
            res[f"shuffle_gap_{condition}"] = shuf_nll / max(tot_tok, 1) - res[f"tf_loss_{condition}"]
            res[f"tf_tok_acc_{condition}"] = tot_correct / max(tot_tok, 1)
            res[f"tf_em_{condition}"] = em / n
            if aux_on:
                res[f"aux_acc_{condition}"] = aux_acc / n
        gen_n = int(self.tc["gen_eval_n"])
        if gen_n > 0:
            rows = self.holdout[:gen_n]
            max_new = int(self.tc["gen_max_new_tokens"])
            # both conditions when they differ, so that K_infer can be chosen on the holdout
            conds = (["train", "deploy"] if int(self.m["K_infer"]) < int(self.m["K_train"])
                     else [self.sc["select_condition"]])
            for cond in conds:
                correct = 0
                for i in range(0, len(rows), bs):
                    chunk = rows[i:i + bs]
                    lat = self._latents(chunk, cond, train=False)
                    prefixes = self.decoder.prefix_embeds([self.task.pre_text(e) for e in chunk], lat["latents"])
                    texts = self.decoder.generate_greedy(prefixes, max_new)
                    correct += sum(bool(s["correct"]) for s in self.task.score_many(texts, chunk))
                res[f"gen_acc_{cond}"] = correct / max(len(rows), 1)
        aug_n = int(self.tc["eval_aug_n"])
        if aug_n > 0 and self.pipe.per_example:
            # the same probes on holdout rows transformed by this stage's pipeline (progress 0)
            rng = random.Random(12345)
            aug = [self.pipe.sample(e, rng, 0.0) for e in self.holdout[:aug_n]]
            nll = tok = shuf = 0.0
            for i in range(0, len(aug), bs):
                chunk = aug[i:i + bs]
                f = self._forward(chunk, "train", train=False)
                t = float(f["ce"]["n_tokens"])
                nll += float(f["ce"]["loss"]) * t
                tok += t
                if len(chunk) > 1:
                    s = self.decoder.ce([self.task.pre_text(e) for e in chunk], f["lat"]["latents"].roll(1, dims=0),
                                        [e.answer for e in chunk], use_prefix_cache=bool(self.tc["prefix_cache"]))
                    shuf += float(s["loss"]) * float(s["n_tokens"])
            res["tf_loss_aug"] = nll / max(tok, 1)
            res["shuffle_gap_aug"] = shuf / max(tok, 1) - res["tf_loss_aug"]
        if was_training:
            mod.train()
        if self.ema is not None:
            self.ema.swap(mod)
        return res

    def selection_value(self, metrics: Dict[str, Any]) -> float:
        """The configured metric, tie-broken by teacher-forced token accuracy."""
        cond = self.sc["select_condition"]
        key = self.sc["select_by"] or f"{self.tc['select_by']}_{cond}"
        if key not in metrics:
            raise KeyError(f"selection metric {key!r} was not computed (train.select_by={self.tc['select_by']}, "
                           f"train.gen_eval_n={self.tc['gen_eval_n']}, train.eval_aug_n={self.tc['eval_aug_n']})")
        return float(metrics[key]) + 1e-3 * float(metrics.get(f"tf_tok_acc_{cond}", 0.0))

    # -- checkpoints ----------------------------------------------------------------------

    def _eval_state(self) -> Dict[str, torch.Tensor]:
        if self.ema is None:
            return {k: v.detach().cpu().clone() for k, v in self.trainable.state_dict().items()}
        self.ema.swap(self.trainable)
        state = {k: v.detach().cpu().clone() for k, v in self.trainable.state_dict().items()}
        self.ema.swap(self.trainable)
        return state

    def save(self, name: str, step: int, metrics: Dict[str, Any], with_opt: bool = False):
        if not self.ctx.main:
            return
        ck = {"stage": self.stage, "step": step, "metrics": metrics, "config": self.cfg.to_dict(),
              "eval_module": self._eval_state(),
              "module": {k: v.detach().cpu().clone() for k, v in self.trainable.state_dict().items()},
              "ema": None if self.ema is None else {k: v.cpu() for k, v in self.ema.shadow.items()}}
        if self.stage == 2:
            ck["stage1_ckpt"], ck["stage1_sha"] = self.stage1_ckpt, self.stage1_sha
        if with_opt:
            ck["opt"] = self.opt.state_dict()
        tmp = os.path.join(self.stage_dir, f".{name}.tmp")
        torch.save(ck, tmp)
        os.replace(tmp, os.path.join(self.stage_dir, name))

    # -- main loop ------------------------------------------------------------------------

    def run(self, bench_steps: int = 0, resume: bool = False) -> Dict[str, Any]:
        ctx, tc = self.ctx, self.tc
        accum = int(tc["grad_accum"])
        n_params = count_params(self.trainable)
        log(self.cfg.describe())
        log(f"stage {self.stage}: trainable params {n_params:,} | pipeline {self.pipe} | pool {len(self.pool)} "
            f"holdout {len(self.holdout)}{' (FIT PROBE: holdout = pool)' if self.is_probe else ''} | world "
            f"{ctx.world} x batch {self.batch_size} x accum {accum} | total steps {self.total_steps} "
            f"(warmup {warmup_steps(self.sc, self.total_steps)})")
        if ctx.main:
            ref = self.refiner or Refiner(self.m, self.decoder.hidden, self.decoder.shell)
            log("parameters (stage 1 + stage 2 modules):\n" + format_breakdown(param_breakdown(self.proposer, ref)))
            with open(os.path.join(self.stage_dir, "run_config.json"), "w") as f:
                json.dump({**self.cfg.to_dict(), "stage": self.stage, "trainable_params": n_params,
                           "pipeline": repr(self.pipe), "pool": len(self.pool), "holdout": len(self.holdout),
                           "world": ctx.world, "total_steps": self.total_steps, "is_probe": self.is_probe}, f, indent=2)
        start_step, best = 1, -float("inf")
        if resume:
            path = os.path.join(self.stage_dir, "last.pt")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.trainable.load_state_dict(ck["module"])
            if ck.get("opt") is not None:
                self.opt.load_state_dict(ck["opt"])
            if self.ema is not None and ck.get("ema") is not None:
                self.ema.shadow = {k: v.to(ctx.device) for k, v in ck["ema"].items()}
            start_step = int(ck["step"]) + 1
            best = float(ck["metrics"].get("best", -float("inf")))
            log(f"resumed from {path} at step {start_step}")
        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)

        self.trainable.train()
        seed = int(tc["seed"])
        eps = float("nan")
        t0 = time.time()
        t_mark, step_mark, ex_count = t0, start_step - 1, 0
        last_metrics: Dict[str, Any] = {}
        last_ce = {"ce": float("nan"), "tok_acc": float("nan"), "em": float("nan")}
        aux_on = float(self.sc["aux_readout"]["weight"]) > 0
        end = self.total_steps if bench_steps <= 0 else start_step + bench_steps - 1
        for step in range(start_step, end + 1):
            lr = lr_at(step, self.total_steps, self.sc)
            for g in self.opt.param_groups:
                g["lr"] = lr
            stats = {"loss": 0.0, "ce": 0.0, "pen": 0.0, "tok_acc": 0.0, "em": 0.0}
            if aux_on:
                stats.update(aux=0.0, aux_acc=0.0)
            progress = (step - 1) / max(1, self.total_steps - 1)
            for micro in range(accum):
                gidx = (step - 1) * accum + micro
                batch = [self.pipe.sample(self.pool[i], random.Random(hash((seed, gidx, i))), progress)
                         for i in self.stream.indices(gidx)]
                sync = micro == accum - 1 or ctx.world == 1
                cm = self.model.no_sync() if (ctx.world > 1 and not sync) else _Null()
                with cm:
                    f = self._forward(batch, "train", train=True, step=step)
                    (f["loss"] / accum).backward()
                if f["ce"] is not None:
                    last_ce = {"ce": float(f["ce"]["loss"].detach()), "tok_acc": float(f["ce"]["token_acc"]),
                               "em": float(f["ce"]["answer_em"])}
                stats["loss"] += float(f["loss"].detach()) / accum
                stats["pen"] += (float(f["penalty"].detach()) if f["penalty"] is not None else 0.0) / accum
                for key in ("ce", "tok_acc", "em"):                  # the latest step that computed the decoder CE
                    stats[key] += last_ce[key] / accum
                if f["aux"] is not None:
                    stats["aux"] += float(f["aux"]["loss"].detach()) / accum
                    stats["aux_acc"] += float(f["aux"]["acc"]) / accum
                ex_count += len(batch)
            gnorm = torch.nn.utils.clip_grad_norm_(self.trainable.parameters(), float(self.sc["grad_clip"]))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.trainable)

            if step % int(tc["log_every"]) == 0 or step == end:
                now = time.time()
                sps = (step - step_mark) / max(now - t_mark, 1e-9)
                eps = ex_count * ctx.world / max(now - t_mark, 1e-9)
                t_mark, step_mark, ex_count = now, step, 0
                eta_h = (self.total_steps - step) / max(sps, 1e-9) / 3600
                rec = {"step": step, "lr": lr, "grad_norm": float(gnorm),
                       **{k: ctx.all_mean(v) for k, v in stats.items()},
                       "cos_L0": mean_pairwise_cos(f["lat"]["L0"].detach()),
                       "norm_L0": float(f["lat"]["L0"].detach().norm(dim=-1).mean()),
                       "steps_per_s": sps, "examples_per_s": eps}
                if f["lat"]["delta"] is not None:
                    rec["norm_delta"] = float(f["lat"]["delta"].detach().norm(dim=-1).mean())
                    rec["cos_delta"] = mean_pairwise_cos(f["lat"]["delta"].detach())
                log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items())
                    + f" eta={eta_h:.2f}h")
                if ctx.main:
                    with open(self.metrics_path, "a") as fh:
                        fh.write(json.dumps({"kind": "train", **rec}) + "\n")

            if bench_steps > 0:
                continue
            if step % int(tc["eval_every"]) == 0 or step == self.total_steps:
                ctx.barrier()
                if ctx.main:
                    te = time.time()
                    last_metrics = self.evaluate()
                    value = self.selection_value(last_metrics)
                    tag = ""
                    if value > best:
                        best = value
                        self.save("best.pt", step, last_metrics)
                        tag = " [new best -> best.pt]"
                    self.save("last.pt", step, {**last_metrics, "best": best}, with_opt=True)
                    log(f"eval step={step} ({time.time() - te:.0f}s) "
                        + " ".join(f"{k}={v:.4f}" for k, v in last_metrics.items()) + tag)
                    with open(self.metrics_path, "a") as fh:
                        fh.write(json.dumps({"kind": "eval", "step": step, **last_metrics, "best": best}) + "\n")
                ctx.barrier()

        summary = {"stage": self.stage, "best_selection": best, "last_eval": last_metrics,
                   "wall_h": (time.time() - t0) / 3600}
        if ctx.device.type == "cuda":
            summary["peak_mem_gib"] = torch.cuda.max_memory_allocated(ctx.device) / 2 ** 30
        if bench_steps > 0:
            summary["bench"] = {"steps": bench_steps, "examples_per_s_last_window": eps}
        log(f"done: {json.dumps(summary)}")
        ctx.close()
        return summary


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
