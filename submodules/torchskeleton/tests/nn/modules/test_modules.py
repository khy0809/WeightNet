# -*- coding: utf-8 -*-
import os
import sys

import torch
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(base_dir)
from skeleton.nn.modules.modules import Mul, Flatten, Concat, MergeSum, Split, DelayedPass


def test_mul():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Mul(0.5)(torch.Tensor(x))
    assert x.shape == y.shape
    assert (x * 0.5 == y.numpy()).all()

    y = Mul(2.0)(torch.Tensor(x))
    assert x.shape == y.shape
    assert (x * 2.0 == y.numpy()).all()


def test_flatten():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Flatten()(torch.Tensor(x))
    assert y.shape == (128, 3 * 32 * 32)


def test_concat():
    x1 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    x2 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Concat()(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 6, 32, 32)

    y = Concat(dim=2)(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 3, 64, 32)

    y = Concat(dim=3)(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 3, 32, 64)


def test_merge_sum():
    x1 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    x2 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = MergeSum()(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 3, 32, 32)
    assert (x1 + x2 == y.numpy()).all()


def test_split():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    m = Split(torch.nn.Sequential(), torch.nn.Identity(), Mul(0.5), Mul(1.5))
    y1, y2, y3, y4 = m(torch.Tensor(x))

    assert y1.shape == (128, 3, 32, 32)
    assert y2.shape == (128, 3, 32, 32)
    assert y3.shape == (128, 3, 32, 32)
    assert y4.shape == (128, 3, 32, 32)

    assert (x == y1.numpy()).all()
    assert (x == y2.numpy()).all()
    assert (x * 0.5 == y3.numpy()).all()
    assert (x * 1.5 == y4.numpy()).all()


def test_delayed_pass():
    x1 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    x2 = np.random.rand(128, 3, 32, 32).astype(np.float32)

    m = DelayedPass()
    y1 = m(torch.Tensor(x1))
    y2 = m(torch.Tensor(x2))
    y3 = m(None)
    assert y1 is None
    assert (x1 == y2.numpy()).all()
    assert (x2 == y3.numpy()).all()
