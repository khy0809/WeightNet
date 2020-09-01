import torch
import math
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import warnings

import torchvision.transforms.functional as F


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def _overlap(target, bboxes, th):
    x1, y1, w, h = target
    x2, y2 = x1 + w, y1 + h
    # area = w * h

    for (xx1, yy1, ww, hh) in bboxes:
        xx2, yy2 = xx1 + ww, yy1 + hh
        area = ww * hh
        lx, ty, rx, by = max(x1, xx1), max(y1, yy1), min(x2, xx2), min(y2, yy2)
        iw, ih = max(0, rx - lx), max(0, by - ty)

        intersection = 1.0 * (iw * ih) / area
        # print(x1, x2, y1, y2, '|', xx1, xx2, yy1, yy2, '|', lx, rx, ty, by, '|', iw, ih, area, intersection)
        if intersection > th:
            return True
    return False


class RandomCropBBox(object):
    def __init__(self, min_object_covered=0.1, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.min_object_covered = min_object_covered
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, bbox, min_object_convered, scale, ratio):
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                for _ in range(10):
                    i = random.randint(0, height - h)
                    j = random.randint(0, width - w)
                    overlapped = _overlap((j, i, w, h), bbox, min_object_convered)
                    if overlapped:
                        return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, bbox=None):
        bbox = bbox if bbox is not None else [[0, 0, img.width, img.height]]
        i, j, h, w = self.get_params(img, bbox, self.min_object_covered, self.scale, self.ratio)
        return F.crop(img, i, j, h, w)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(min_object_covered={0}'.format(self.min_object_covered)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0})'.format(tuple(round(r, 4) for r in self.ratio))
        return format_string
