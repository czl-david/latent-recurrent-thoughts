"""Task registry: each task config names its data module (data.module: package.module:Class), so no
code outside a task's own module refers to a specific task."""

import importlib

from lrt.data.base import Example, Pipeline, Task


def get_task(cfg) -> Task:
    mod_name, cls_name = cfg["data"]["module"].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    assert issubclass(cls, Task), f"{cfg['data']['module']} is not a Task"
    return cls(cfg)
