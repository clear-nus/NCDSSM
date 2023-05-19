import re
from argparse import ArgumentParser, ArgumentTypeError
from torch import nn
import yaml


import os
from datetime import datetime

from ncdssm.type import Dict


def get_config_and_setup_dirs(filename: str = "config.yaml"):
    with open(filename, "r") as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config["exp_root_dir"] = config["exp_root_dir"].format(
        model=config["model"].lower(),
        dataset=config["dataset"].lower(),
        timestamp=timestamp,
    )
    config["log_dir"] = os.path.join(config["exp_root_dir"], "logs")
    config["ckpt_dir"] = os.path.join(config["exp_root_dir"], "ckpts")
    os.makedirs(config["log_dir"])
    os.makedirs(config["ckpt_dir"])

    return config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def add_config_to_argparser(config: Dict, parser: ArgumentParser):
    for k, v in config.items():
        sanitized_key = re.sub(r"[^\w\-]", "", k).replace("-", "_")
        val_type = type(v)
        if val_type not in {int, float, str, bool}:
            print(f"WARNING: Skipping key {k}!")
            continue
        if val_type == bool:
            parser.add_argument(f"--{sanitized_key}", type=str2bool, default=v)
        else:
            parser.add_argument(f"--{sanitized_key}", type=val_type, default=v)
    return parser


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    elif name == "softplus":
        return nn.Softplus
    elif name == "relu":
        return nn.ReLU
    elif name == "softmax":
        return nn.Softmax
    else:
        raise ValueError(f"Unknown non-linearity {name}")


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val
