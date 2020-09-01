import logging

import torch


LOGGER = logging.getLogger(__name__)


class GradualWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, steps, multiplier, start_from_zero=False, last_epoch=-1):
        self.steps = steps
        self.multiplier = multiplier
        self.start_from_zero = start_from_zero

        super(GradualWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.start_from_zero:
            multiplier = self.multiplier * min(1.0, (self.last_epoch / self.steps))
        else:
            multiplier = 1 + ((self.multiplier - 1) * min(1.0, (self.last_epoch / self.steps)))
        return [lr * multiplier for lr in self.base_lrs]


class Schedulers:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self, epoch=None):
        _ = [s.step(epoch=epoch) for s in self.schedulers]

    def get_last_lr(self):
        return self.schedulers[-1].get_last_lr()

    def state_dict(self):
        return [s.state_dict() for s in self.schedulers]

    def load_state_dict(self, state_dict):
        for s, state in zip(self.schedulers, state_dict):
            s.load_state_dict(state)

