# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch


LOGGER = logging.getLogger(__name__)


class CrossEntropyVector(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyVector, self).__init__()
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        if input.shape != target.shape:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.to(device=input.device, dtype=torch.float, non_blocking=True)

        log_probs = self.logsoftmax(input)
        loss = (-target * log_probs)

        if self.reduction in ['avg', 'mean']:
            loss = loss.mean(0).sum()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon, reduction='mean'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets.detach() * log_probs)

        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class WeightedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, positive_weight=1.0, negative_weight=1.0, weight=None, reduction='mean', pos_weight=None):
        super(WeightedBCEWithLogitsLoss, self).__init__(weight=weight, reduction='none', pos_weight=pos_weight)
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.final_reduction = reduction

    def forward(self, input, target):
        loss = super(WeightedBCEWithLogitsLoss, self).forward(input, target)

        trues = (target.detach() > 0.5).float()
        falses = (1. - trues)
        loss1 = (loss * trues * self.positive_weight)
        loss2 = (loss * falses * self.negative_weight)
        if self.final_reduction == 'mean':
            loss = (loss1 + loss2).mean()
        else:
            loss = loss1 + loss2
        return loss


class WeightedLabelSmoothBCEWithLogitsLoss(WeightedBCEWithLogitsLoss):
    def __init__(self, epsilon, positive_weight=1.0, negative_weight=1.0, weight=None, reduction='mean', pos_weight=None):
        super(WeightedLabelSmoothBCEWithLogitsLoss, self).__init__(positive_weight, negative_weight, weight, reduction, pos_weight)
        self.epsilon = epsilon

    def forward(self, input, target):
        # from: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/losses/losses_impl.py#L703
        target = ((1 - self.epsilon) * target) + (0.5 * self.epsilon)
        return super(WeightedLabelSmoothBCEWithLogitsLoss, self).forward(input, target)
