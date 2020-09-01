# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math

import torch
from torch import distributed as dist
from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes


class DistributedWeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).
    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """
    def __init__(self, weights, num_samples, replacement=True, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(math.ceil(num_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = torch.multinomial(self.weights, self.num_samples * self.num_replicas,
                                    self.replacement, generator=g).tolist()

        # subsample
        indices = indices[self.rank:(self.num_samples * self.num_replicas):self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
