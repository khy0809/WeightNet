# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import shutil
import random
import logging

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(path, state, is_best=False, filename='checkpoint.pth.tar'):
    os.makedirs(path, exist_ok=True)
    torch.save(state, '%s/%s' % (path, filename))

    if is_best:
        shutil.copyfile(filename, '%s/%s' % (path, 'best.pth.tar'))


def save_checkpoints(epoch, path, state, is_best=False, keep_last=30):
    os.makedirs(path, exist_ok=True)
    torch.save(state, '%s/%s' % (path, 'last.pth.tar'))
    shutil.copyfile('%s/%s' % (path, 'last.pth.tar'), '%s/%s' % (path, '%08d.pth.tar' % epoch))

    if is_best:
        shutil.copyfile('%s/%s' % (path, 'last.pth.tar'), '%s/%s' % (path, 'best.pth.tar'))

    if os.path.exists('%s/%08d.pth.tar' % (path, epoch - keep_last)):
        os.remove('%s/%08d.pth.tar' % (path, epoch - keep_last))


class Saver:
    def __init__(self, path, keep_last=30, check_evaluate_key=['metrics', 'valid', 'accuracy']):
        self.path = path
        self.keep_last = keep_last
        self.check_evaluate_key = check_evaluate_key
        self.best_score = 0.0
        os.makedirs(path, exist_ok=True)

    def save(self, epoch, state):
        torch.save(state, '%s/%s' % (self.path, 'last.pth.tar'))
        shutil.copyfile('%s/%s' % (self.path, 'last.pth.tar'), '%s/%s' % (self.path, '%08d.pth.tar' % epoch))

        score = state
        for k in self.check_evaluate_key:
            score = score[k] if k in score else 0.0

        if self.best_score < score:
            shutil.copyfile('%s/%s' % (self.path, 'last.pth.tar'), '%s/%s' % (self.path, 'best.pth.tar'))
            LOGGER.info('[Saver] update best score:%.4f (before:%.4f)', score, self.best_score)
        self.best_score = max(self.best_score, score)

        if os.path.exists('%s/%08d.pth.tar' % (self.path, epoch - self.keep_last)):
            os.remove('%s/%08d.pth.tar' % (self.path, epoch - self.keep_last))






