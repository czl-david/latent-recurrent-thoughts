"""Model, training, evaluation and tooling code is data-agnostic: task-specific code lives only in lrt/data/
(the data layer) and configs/. A task or dataset name used as an identifier or string literal anywhere else
(comments and docstrings aside) fails this test."""

import ast
import glob
import importlib
import os
import re

from lrt import config as C
from tests.conftest import REPO, exp

DATASET_WORDS = {"sudoku", "countdown", "cd4", "mbpp", "humaneval", "strategyqa", "sqa", "gsm8k"}


def banned() -> set:
    return {t.lower() for t in C.available_tasks()} | DATASET_WORDS


def _docstring_ids(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _words(s: str):
    return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", s) if w}


def test_core_python_is_task_agnostic():
    words = banned()
    offenders = []
    for path in sorted(glob.glob(os.path.join(REPO, "lrt", "*.py"))):
        tree = ast.parse(open(path).read())
        docs = _docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                hit = _words(node.value) & words
            elif isinstance(node, ast.Name):
                hit = _words(node.id) & words
            elif isinstance(node, ast.Attribute):
                hit = _words(node.attr) & words
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                hit = _words(node.name) & words
            else:
                continue
            if hit:
                offenders.append(f"{os.path.relpath(path, REPO)}:{getattr(node, 'lineno', '?')} {sorted(hit)}")
    assert not offenders, "task-specific code outside lrt/data/:\n" + "\n".join(offenders)


def test_scripts_are_task_agnostic():
    words = banned()
    offenders = []
    for path in sorted(glob.glob(os.path.join(REPO, "scripts", "*.sh"))):
        for i, line in enumerate(open(path), 1):
            if _words(line.split("#", 1)[0]) & words:
                offenders.append(f"{os.path.relpath(path, REPO)}:{i}: {line.strip()}")
    assert not offenders, "task names in script logic:\n" + "\n".join(offenders)


def test_every_task_config_names_an_importable_data_module():
    for task in C.available_tasks():
        mod, cls = C.load(exp(task, "default"))["data"]["module"].split(":")
        assert mod.startswith("lrt.data."), mod
        assert hasattr(importlib.import_module(mod), cls)
