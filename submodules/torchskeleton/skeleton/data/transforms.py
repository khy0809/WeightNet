# -*- coding: utf-8 -*-
import os
import logging
import shutil
import hashlib
import random

import numpy as np
import torch
from PIL import Image
import torchvision


LOGGER = logging.getLogger(__name__)


class Pad:
    def __init__(self, border, mode='reflect'):
        self.border = border
        self.mode = mode

    def __call__(self, image):
        img = np.pad(image, [(self.border, self.border), (self.border, self.border), (0, 0)], mode=self.mode)
        return Image.fromarray(img)


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = torchvision.transforms.Compose(self.transforms)
        return transform(img)


class Lighting:
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Cutout:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        h, w = image.size(1), image.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.choice(range(h))
        x = np.random.choice(range(w))

        y1 = np.clip(y - self.height // 2, 0, h)
        y2 = np.clip(y + self.height // 2, 0, h)
        x1 = np.clip(x - self.width // 2, 0, w)
        x2 = np.clip(x + self.width // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask).to(device=image.device, dtype=image.dtype)
        mask = mask.expand_as(image)
        image *= mask
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(height={0}, width={1})'.format(self.height, self.width)


class TensorRandomHorizontalFlip:
    def __call__(self, tensor):
        choice = np.random.choice([True, False])
        return torch.flip(tensor, dims=[-1]) if choice else tensor


class TensorRandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, tensor):
        C, H, W = tensor.shape
        h = np.random.choice(range(H + 1 - self.height))
        w = np.random.choice(range(W + 1 - self.width))
        return tensor[:, h:h+self.height, w:w+self.width]


class ImageWriter:
    def __init__(self, root, delete_folder_exists=True):
        self.root = root

        if delete_folder_exists and os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

    def __call__(self, image):
        filename = hashlib.md5(image.tobytes()).hexdigest()
        filepath = os.path.join(self.root,  filename + '.jpg')
        with open(filepath, 'wb') as f:
            image.save(f, format='jpeg')
        return image
