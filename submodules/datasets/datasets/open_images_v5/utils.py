import numpy as np
from torchvision.transforms import Compose
from torchvision.datasets.vision import StandardTransform


class TargetToVector(object):
    """Convert target class to vector. """
    classes: 'OrderedDict[str, str]'
    _mid_to_idx: dict

    def __init__(self, classes=None):
        if classes is not None:
            self.set_classes(classes)

    def set_classes(self, classes: 'OrderedDict[str, str]'):
        self.classes = classes
        self._mid_to_idx = {mid: idx for idx, mid in enumerate(self.classes)}

    def __call__(self, target):
        positive_ids, negative_ids = target

        labels = np.full(len(self.classes), -1., dtype=np.float)
        pos_indicies = [self._mid_to_idx[mid] for mid in positive_ids
                        if mid in self.classes]
        labels[pos_indicies] = 1.
        neg_indicies = [self._mid_to_idx[mid] for mid in negative_ids
                        if mid in self.classes]
        labels[neg_indicies] = 0.

        return labels

    def __repr__(self):
        return self.__class__.__name__ + '(num_class={})'.format(len(self.classes))

    @staticmethod
    def inject_classes(transforms, classes):
        if isinstance(transforms, TargetToVector):
            transforms.set_classes(classes)
        elif isinstance(transforms, StandardTransform):
            TargetToVector.inject_classes(transforms.target_transform, classes)
        elif isinstance(transforms, Compose):
            for transform in transforms.transforms:
                TargetToVector.inject_classes(transform, classes)
