import os
import random
import time
from contextlib import contextmanager
from glob import glob
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class HParams(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError('Provide at most 1 positional argument')

        if len(args) == 0:
            super().__init__(**kwargs)
        else:
            if isinstance(args[0], dict):
                super().__init__(**args[0])
            else:
                super().__init__(**vars(args[0]))
            self.add(**kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return (key in self.__dict__)

    def add(self, *args, overwrite: bool = False, **kwargs):
        if len(args) == 0:
            for k, v in kwargs.items():
                if k in self.__dict__ and not overwrite:
                    raise ValueError(f'Duplicate key "{k}" found while adding!')
                self.__dict__[k] = v
            return self

        for hp in args:
            if isinstance(hp, dict):
                self.add(**hp, overwrite=overwrite)
            else:
                self.add(**vars(hp), overwrite=overwrite)

        self.add(**kwargs, overwrite=overwrite)

        return self

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def clone(self):
        return HParams(self)

    @classmethod
    def load(cls, fn, fmt='json'):
        if fmt == 'json':
            import json
            with open(fn, 'r') as f:
                hp = cls(**json.load(f))
            return hp
        else:
            raise NotImplementedError(f'Unsupported format "{fmt}"')

    def save(self, fn, fmt='json'):
        if fmt == 'json':
            import json
            with open(fn, 'w') as f:
                json.dump(self.__dict__, f, indent=2)
        else:
            raise NotImplementedError(f'Unsupported format "{fmt}"')


class Logger:
    def __init__(self, log_dir):
        if os.path.exists(log_dir):
            assert len(glob('events.out.tfevents.*')) == 0, (
                f'Tensorboard log already exists in {log_dir}')
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

    def flush(self):
        self.writer.flush()

    def log_scalar(self, tag, val, step):
        if hasattr(val, 'item'):
            val = val.item()
        self.writer.add_scalar(tag, val, global_step=step)

    def log_scalars(self, tag_value_dict, step: int):
        for tag, val in tag_value_dict.items():
            self.log_scalar(tag, val, step)

    def log_image(self, tag: str, img: torch.Tensor, step: int):
        assert img.ndim == 3
        self.writer.add_image(tag, img, global_step=step, dataformats='CHW')

    def log_image_grid(self, tag, imgs, step: int, **kwargs):
        assert imgs.ndim == 4
        img_grid = make_grid(imgs, **kwargs)
        self.log_image(tag, img_grid, step)

    def add_graph(self, *args, **kwargs):
        self.writer.add_graph(*args, **kwargs)



def check_or_mkdir(path):
    if os.path.lexists(path):
        if os.path.isdir(path) and len(os.listdir(path)) == 0:
            print(f'Directory {path} already exists, but is empty.')
        else:
            raise ValueError(f'Directory {path} is not empty! Terminating.')
    else:
        print(f'Creating directory {path}')
        os.makedirs(path)


def checkpoint(out_dir: str, dump_dict: Dict, *,
               step: int = None, epoch: int = None, prefix='ckpt', keep_every: int = 10000):
    assert (step is None) ^ (epoch is None), 'Exactly one of `step` or `epoch` should be given'
    assert os.path.isdir(out_dir), f'Invalid output directory: {out_dir}'

    if step is not None and step >= 10 ** 7:
        print(f'[WARNING] step {step} too large!')
    if epoch is not None and epoch >= 10 ** 4:
        print(f'[WARNING] epoch {epoch} too large!')

    unit, idx = ('step', step) if epoch is None else ('epoch', epoch)

    # Delete unnecessary checkpoints
    if epoch is not None:
        ckpts = get_checkpoints(out_dir)
        if len(ckpts) > 0:
            prev_idx = int(os.path.basename(ckpts[-1])[len(f'{prefix}_epoch='):-3])
            if prev_idx % keep_every != 0:
                os.remove(ckpts[-1])

    # Checkpoint the current state
    if epoch is not None:
        torch.save(dump_dict, os.path.join(out_dir, f'{prefix}_{unit}={idx:04d}.pt'))
    else:
        torch.save(dump_dict, os.path.join(out_dir, f'{prefix}_{unit}={idx:07d}.pt'))


def get_checkpoints(out_dir: str, prefix: str = 'ckpt', unit='epoch'):
    fns = glob(os.path.join(out_dir, f'{prefix}_*.pt'))
    if len(fns) == 0:
        return []

    # TODO: do proper parsing and extract `unit`
    idx = len(f'{prefix}_{unit}=')
    fns.sort(key=lambda fn: int(os.path.basename(fn)[idx:-3]))

    return fns


def get_param_count(model: nn.Module, requires_grad=True):
    if requires_grad:
        counts = [np.prod(p.shape).item() for p in model.parameters() if p.requires_grad]
    else:
        counts = [np.prod(p.shape).item() for p in model.parameters()]

    return sum(counts)


@contextmanager
def timed(desc: str, verbose=True) -> None:
    if verbose:
        print(f'{desc} ', end='', flush=True)
    start = time.time()
    yield
    elapsed = time.time() - start
    if verbose:
        print(f'done ({elapsed:.2f} sec)')


# Copied from pytorch-lightning
def seed_everything(seed: int) -> int:
    """Function that sets seed for pseudo-random number generators  in:
        pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed > max_seed_value) or (seed < min_seed_value):
        raise ValueError(
            f'Seed value ({seed}) is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}')

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def print_model_info(model: torch.nn.Module):
    cnt_train = sum(np.prod(p.shape) for p in model.parameters() if p.requires_grad)
    cnt_total = sum(np.prod(p.shape) for p in model.parameters())
    print(f'Model has {cnt_train} trainable parameters ({cnt_total} total). Training = {model.training}')


def test_hparams():
    def check(target):
        if isinstance(target, HParams):
            assert vars(target) == {'a': 10, 'b': 'b'}
        elif isinstance(target, dict):
            assert target == {'a': 10, 'b': 'b'}
        else:
            raise ValueError

    # Construction by kwargs
    hp_kwargs = HParams(a=10, b='b')
    check(hp_kwargs)

    # Construction by copying
    check(HParams(hp_kwargs))

    # Construction with dict
    check(HParams({'a': 10, 'b': 'b'}))

    # Construction with additional arg
    hp_partial = HParams(a=10)
    check(HParams(hp_partial, b='b'))

    # Using .items()
    check({k: v for k, v in hp_kwargs.items()})
