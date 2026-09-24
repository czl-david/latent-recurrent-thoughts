"""Configuration.

A run is described by one experiment file, e.g. configs/experiments/<task>/<name>.yaml:

    task: <task>                        # also loads configs/tasks/<task>.yaml
    extends: [../../presets/x.yaml]     # optional preset files, merged in order (paths relative to this file)
    run_name: <name>                    # optional; default: the experiment file name without .yaml
    <sections of configs/base.yaml>     # overrides

Effective configuration (later wins):

    configs/base.yaml  <-  configs/tasks/<task>.yaml  <-  extends  <-  experiment body  <-  --set key=value

Dictionaries merge recursively; lists and scalars replace. After the task file, every key must
already exist (a typo fails loudly), except that new named pipelines may be added under
data.pipelines. Runs live in $LRT_OUT_ROOT/<task>/<run_name>/.
"""

import copy
import os
from typing import Any, Dict, Iterable, List, Optional

import yaml

from lrt.paths import configs_dir, out_root

OPEN_SECTIONS = ("data.pipelines",)          # sections where new keys may be introduced


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: Dict[str, Any], over: Dict[str, Any], strict: bool, where: str = "") -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        here = f"{where}.{key}" if where else key
        if key not in out:
            if strict:
                raise KeyError(f"unknown config key {here!r} (every key must exist in configs/base.yaml or the "
                               f"task file; new pipelines go under data.pipelines)")
            out[key] = copy.deepcopy(value)
        elif isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value, strict and here not in OPEN_SECTIONS, here)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_set(item: str):
    """'a.b.c=value' -> ('a.b.c', value); the value is parsed as YAML ("1e-4" becomes a float)."""
    if "=" not in item:
        raise ValueError(f"--set expects key=value, got {item!r}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
    return key.strip(), value


def assign(tree: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for i, p in enumerate(parts[:-1]):
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"--set {dotted}: no section {'.'.join(parts[:i + 1])!r}")
        node = node[p]
    parent = ".".join(parts[:-1])
    if parts[-1] not in node and parent not in OPEN_SECTIONS:
        raise KeyError(f"--set {dotted}: unknown key (typo?)")
    node[parts[-1]] = value


def validate(t: Dict[str, Any]) -> None:
    m, r = t["model"], t["model"]["refiner"]
    for key in ("d_prime", "heads", "K_train", "K_infer"):
        if not isinstance(m[key], int) or m[key] <= 0:
            raise ValueError(f"model.{key} must be a positive int, got {m[key]!r}")
    if m["K_infer"] > m["K_train"]:
        raise ValueError(f"model.K_infer={m['K_infer']} > model.K_train={m['K_train']}: the proposer only emits "
                         f"K_train latents")
    if m["d_prime"] % m["heads"]:
        raise ValueError(f"model.d_prime={m['d_prime']} is not divisible by model.heads={m['heads']}")
    if m["block_norm"] not in ("pre", "post"):
        raise ValueError(f"model.block_norm must be pre|post, got {m['block_norm']!r}")
    for key in ("S", "H", "T"):
        if not isinstance(r[key], int) or r[key] <= 0:
            raise ValueError(f"model.refiner.{key} must be a positive int, got {r[key]!r}")
    if r["init_state"] not in ("learned", "fixed"):
        raise ValueError(f"model.refiner.init_state must be learned|fixed, got {r['init_state']!r}")
    if r["init_state_shape"] not in ("per_slot", "shared"):
        raise ValueError(f"model.refiner.init_state_shape must be per_slot|shared, got {r['init_state_shape']!r}")
    if float(m["lambda_delta"]) < 0:
        raise ValueError("model.lambda_delta must be >= 0")
    tr = t["train"]
    if tr["precision"] not in ("bf16", "fp32"):
        raise ValueError(f"train.precision must be bf16|fp32, got {tr['precision']!r}")
    if tr["prefix_cache"] and tr["grad_ckpt"]:
        raise ValueError("train.prefix_cache and train.grad_ckpt are mutually exclusive (Hugging Face drops the "
                         "cache inside checkpointed layers)")
    for s in (1, 2):
        aux = t[f"stage{s}"]["aux_readout"]
        if float(aux["weight"]) < 0 or int(aux["ce_every"]) < 1 or int(aux["ce_batch"]) < 0:
            raise ValueError(f"stage{s}.aux_readout: weight >= 0, ce_every >= 1, ce_batch >= 0 required")
    pipes = t["data"]["pipelines"]
    for where, name in (("data.pipeline", t["data"]["pipeline"]), ("stage1.pipeline", t["stage1"]["pipeline"]),
                        ("stage2.pipeline", t["stage2"]["pipeline"])):
        if name is not None and name not in pipes:
            raise ValueError(f"{where}={name!r} is not defined under data.pipelines ({sorted(pipes)})")
    for s in (1, 2):
        sc = t[f"stage{s}"]
        if sc["schedule"] not in ("cosine", "constant"):
            raise ValueError(f"stage{s}.schedule must be cosine|constant, got {sc['schedule']!r}")
        if len(sc["betas"]) != 2:
            raise ValueError(f"stage{s}.betas must have two values")
        if not 0 <= float(sc["warmup_frac"]) < 1:
            raise ValueError(f"stage{s}.warmup_frac must be in [0, 1)")
        if sc["select_condition"] not in ("train", "deploy"):
            raise ValueError(f"stage{s}.select_condition must be train|deploy")
    if t["stage2"]["stage1_ckpt"] not in ("best", "last"):
        raise ValueError("stage2.stage1_ckpt must be best|last")


class Config:
    """The effective configuration of one run (a nested dict plus its provenance)."""

    def __init__(self, tree: Dict[str, Any], task: str, name: str, source: Optional[str], sets: Dict[str, Any]):
        self.tree, self.task, self.name, self.source, self.sets = tree, task, name, source, sets

    def __getitem__(self, key: str) -> Any:
        return self.tree[key]

    def stage(self, n: int) -> Dict[str, Any]:
        return self.tree[f"stage{n}"]

    def pipeline_name(self, stage: int) -> str:
        return self.tree[f"stage{stage}"]["pipeline"] or self.tree["data"]["pipeline"]

    def run_dir(self, name: Optional[str] = None) -> str:
        return os.path.join(out_root(), self.task, name or self.name)

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "name": self.name, "source": self.source, "sets": self.sets,
                "config": self.tree}

    def batch_size(self, stage: int) -> int:
        return int(self.tree[f"stage{stage}"]["batch_size"] or self.tree["train"]["batch_size"])

    def describe(self) -> str:
        m, r, tr = self.tree["model"], self.tree["model"]["refiner"], self.tree["train"]

        def aux(s: int) -> str:
            a = self.tree[f"stage{s}"]["aux_readout"]
            return (f"on(weight={a['weight']}, scale={a['scale']}, ce_batch={a['ce_batch']}, ce_every={a['ce_every']})"
                    if float(a["weight"]) > 0 else "off")

        return (f"{self.task}/{self.name}: d'={m['d_prime']} K_train={m['K_train']} K_infer={m['K_infer']} "
                f"S,H,T={r['S']},{r['H']},{r['T']} lambda={m['lambda_delta']} norm={m['block_norm']} "
                f"init_state={r['init_state']}/{r['init_state_shape']} | pipelines: stage1={self.pipeline_name(1)} "
                f"stage2={self.pipeline_name(2)} | batch: stage1={self.batch_size(1)} stage2={self.batch_size(2)} | "
                f"precision={tr['precision']} grad_ckpt={tr['grad_ckpt']} prefix_cache={tr['prefix_cache']} | "
                f"aux_readout: stage1={aux(1)} stage2={aux(2)}")


def available_tasks() -> List[str]:
    d = os.path.join(configs_dir(), "tasks")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))


def experiment_files() -> List[str]:
    root = os.path.join(configs_dir(), "experiments")
    out = []
    for dirpath, _, files in os.walk(root):
        out += [os.path.join(dirpath, f) for f in files if f.endswith(".yaml")]
    return sorted(out)


def load(path: str, sets: Iterable[str] = ()) -> Config:
    exp = read_yaml(path)
    if "task" not in exp:
        raise ValueError(f"{path}: an experiment file must name its task (task: <name>)")
    task = exp["task"]
    task_file = os.path.join(configs_dir(), "tasks", f"{task}.yaml")
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"{path}: no task config {task_file} (tasks: {available_tasks()})")
    tree = deep_merge(read_yaml(os.path.join(configs_dir(), "base.yaml")), read_yaml(task_file), strict=False)
    here = os.path.dirname(os.path.abspath(path))
    for ext in exp.get("extends") or []:
        tree = deep_merge(tree, read_yaml(os.path.normpath(os.path.join(here, ext))), strict=True)
    body = {k: v for k, v in exp.items() if k not in ("task", "extends", "run_name")}
    tree = deep_merge(tree, body, strict=True)
    applied: Dict[str, Any] = {}
    for item in sets:
        key, value = parse_set(item)
        assign(tree, key, value)
        applied[key] = value
    validate(tree)
    name = exp.get("run_name") or os.path.splitext(os.path.basename(path))[0]
    return Config(tree, task, name, os.path.abspath(path), applied)
