# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch


LOGGER = logging.getLogger(__name__)


class AccuracyMany(torch.nn.Module):
    def __init__(self, topk=(1,)):
        super(AccuracyMany, self).__init__()
        self.topk = topk

    def forward(self, output, target):
        with torch.no_grad():
            if output.shape == target.shape:
                _, target = target.max(-1)
            maxk = max(self.topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in self.topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0 / batch_size))
        return res


class Accuracy(AccuracyMany):
    def __init__(self, topk=1, scale=1.0):
        super(Accuracy, self).__init__((topk,))
        self.scale = scale

    def forward(self, output, target):
        res = super(Accuracy, self).forward(output, target)
        return res[0] * self.scale


def to_onehot(labels, shape):
    onehot = torch.zeros(*shape)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    return onehot


class Fscore(torch.nn.Module):
    def __init__(self, threshold=0.5, beta=1, eps=1e-9):
        super(Fscore, self).__init__()
        self.threshold = threshold
        self.beta = beta
        self.eps = eps

    def forward(self, output, target):
        with torch.no_grad():
            beta2 = self.beta ** 2

            if output.shape != target.shape and target.dtype == torch.long:
                target = to_onehot(target, output.shape).to(device=target.device)

            y_pred = torch.ge(output.float(), self.threshold).float()
            y_true = target.float()

            true_positive = (y_pred * y_true).sum(dim=1)
            precision = true_positive.div(y_pred.sum(dim=1).add(self.eps))
            recall = true_positive.div(y_true.sum(dim=1).add(self.eps))

        return {
            'fscore': torch.mean((precision * recall).div(precision.mul(beta2) + recall + self.eps).mul(1 + beta2)),
            'precision': torch.mean(precision),
            'recall': torch.mean(recall)
        }


class MultiLabel(torch.nn.Module):
    def __init__(self, threshold=0.5, activation=torch.sigmoid, epsilon=1e-8):
        super(MultiLabel, self).__init__()
        self.activation = activation
        self.epsilon = epsilon
        self.confusion = ConfusionMatrix(threshold=threshold, activation=None, epsilon=epsilon)

    def forward(self, output, target):
        with torch.no_grad():
            if self.activation is not None:
                output = self.activation(output)
            if output.shape == target.shape:
                pass
            elif target.dtype in [torch.int32, torch.int64, torch.long]:
                target_onehot = torch.zeros_like(output)
                target_onehot.scatter_(1, target.view(-1, 1), 1)
                target = target_onehot
            else:
                raise ValueError('not support target type(%s) at not matched shape(%s, %s)', target.dtype, output.shape, target.shape)

            nbatch, nclasses = output.shape
            zero_one = torch.all(output == target, dim=1).sum() / nbatch

            tp, tn, fp, fn = self.confusion(output, target, dim=0)
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1score = 2.0 * precision * recall / (precision + recall + self.epsilon)
            precision_per_class = precision.mean()
            recall_per_class = recall.mean()
            f1score_per_class = f1score.mean()

            tp, tn, fp, fn = tp.sum(), tn.sum(), fp.sum(), fn.sum()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1score = 2.0 * precision * recall / (precision + recall + self.epsilon)

        return {
            'zero-one': zero_one,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1score': f1score,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1score_per_class': f1score_per_class,
        }


class AUC(torch.nn.Module):
    def __init__(self, step=30, min_=0.0, max_=1.0, activation=torch.sigmoid, epsilon=1e-8):
        super(AUC, self).__init__()
        self.step = step
        self.min_ = min_
        self.max_ = max_
        self.activation = activation
        self.epsilon = epsilon

    def forward(self, output, target):
        if self.activation is not None:
            output = self.activation(output)
        if output.shape == target.shape:
            pass
        elif target.dtype in [torch.int32, torch.int64, torch.long]:
            target_onehot = torch.zeros_like(output)
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            target = target_onehot
        else:
            raise ValueError('not support target type(%s) at not matched shape(%s, %s)', target.dtype, output.shape, target.shape)

        trues = (target == 1).float()
        falses = 1.0 - trues

        tpr_prev = torch.zeros(1, dtype=output.dtype, device=output.device)[0]
        fpr_prev = torch.zeros(1, dtype=output.dtype, device=output.device)[0]
        area = torch.zeros(1, dtype=output.dtype, device=output.device)[0]
        for s in range(self.step):
            ratio = (self.step - s) / self.step
            th = ratio * (self.max_ - self.min_) + self.min_
            prediction = (output >= th).float()
            tpr = (prediction * trues).sum() / (trues.sum() + self.epsilon)
            fpr = (prediction * falses).sum() / (falses.sum() + self.epsilon)

            fpr_min, fpr_max = min(fpr, fpr_prev), max(fpr, fpr_prev)
            tpr_min, tpr_max = min(tpr, tpr_prev), max(tpr, tpr_prev)
            width = fpr_max - fpr_min
            area += (width * tpr_min) + (0.5 * width * (tpr_max - tpr_min))
            tpr_prev, fpr_prev = tpr, fpr
        return area


class ConfusionMatrix(torch.nn.Module):
    def __init__(self, threshold=0.5, activation=torch.sigmoid, epsilon=1e-8):
        super(ConfusionMatrix, self).__init__()
        self.threshold = threshold
        self.activation = activation
        self.epsilon = epsilon

    def forward(self, output, target, dim=None):
        if self.activation is not None:
            output = self.activation(output)
        if output.shape == target.shape:
            pass
        elif target.dtype in [torch.int32, torch.int64, torch.long]:
            target_onehot = torch.zeros_like(output)
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            target = target_onehot
        else:
            raise ValueError('not support target type(%s) at not matched shape(%s, %s)', target.dtype, output.shape, target.shape)

        prediction = (output >= self.threshold).to(dtype=target.dtype)

        trues = (target == 1).float()
        falses = (1 - trues)
        tp = (prediction * trues).sum(dim=dim) + self.epsilon
        fp = (prediction * falses).sum(dim=dim) + self.epsilon
        tn = ((1-prediction) * falses).sum(dim=dim) + self.epsilon
        fn = ((1-prediction) * trues).sum(dim=dim) + self.epsilon
        return tp, tn, fp, fn
