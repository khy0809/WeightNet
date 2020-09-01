from __future__ import print_function
import xmltodict
import os

from .imagenet import ImageNet
from .imagenet_transform import RandomCropBBox


class ImageNetLoc(ImageNet):
    def __init__(self, split='train', **kwargs):
        super(ImageNetLoc, self).__init__(split, **kwargs)

    def __getitem__(self, index, with_transform=True):
        path, target = self.samples[index]
        sample = self.loader(path)

        meta_path = path.replace('/%s/' % self.split, '/%s_loc/' % self.split).replace('JPEG', 'xml')
        if os.path.exists(meta_path):
            meta = xmltodict.parse(open(meta_path).read())['annotation']
            objs = meta['object'] if isinstance(meta['object'], list) else [meta['object']]
            bboxs = [
                [
                    int(o['bndbox']['xmin']),
                    int(o['bndbox']['ymin']),
                    int(o['bndbox']['xmax']) - int(o['bndbox']['xmin']),
                    int(o['bndbox']['ymax']) - int(o['bndbox']['ymin'])
                ] for o in objs
            ]
        else:
            bboxs = [
                [0, 0, sample.width, sample.height]
            ]

        if self.transforms is not None and with_transform:
            sample, target = self.transforms(sample, target)
        return sample, bboxs, target


class ImageNetLocCovered(ImageNetLoc):
    def __init__(self, split='train', special_transform=RandomCropBBox(), **kwargs):
        super(ImageNetLocCovered, self).__init__(split, **kwargs)
        self.special_transform = special_transform

    def __getitem__(self, index):
        sample, bboxs, target = super(ImageNetLocCovered, self).__getitem__(index, with_transform=False)
        sample = self.special_transform(sample, bboxs)
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)
        return sample, target


class ImageNetCovered(ImageNet):
    def __init__(self, split='train', special_transform=RandomCropBBox(), **kwargs):
        super(ImageNetCovered, self).__init__(split, **kwargs)
        self.special_transform = special_transform

    def __getitem__(self, index):
        sample, target = super(ImageNetCovered, self).__getitem__(index, with_transform=False)
        sample = self.special_transform(sample)
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)
        return sample, target
