import os
import csv
from collections import namedtuple, OrderedDict, defaultdict
from functools import lru_cache
from typing import List

from torchvision.datasets import VisionDataset
from PIL import Image


class TencentML7M(VisionDataset):
    """TencentML -

    Args:
        phase (str): One of (train, validation, test)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    Attributes:
        classes: List[[labels]]
        synsets: List[synset_id]
        hierarchy: List[parent_idx]
    """
    root = '/data/public/rw/datasets/tencent-ml'
    phases = {'train', 'validation', 'test'}

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None, load_pil=True, lang='en', use_subset=False):
        super().__init__(root=self.root,
                         transforms=transforms, transform=transform, target_transform=target_transform)
        assert split in self.phases, f'{split} is not in {self.phases}'
        self.split = split
        self._load_pil = load_pil
        self._use_subset = use_subset
        self.classes, self.synsets, self.hierarchy, self.subsets = self.get_classes(lang, use_subset)
        self.annotations = self._load_annotations()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, (positive_labels, negative_labels))
        """
        dataset_type: str
        img_id: str
        labels: List[int]
        dataset_type, img_id, labels = self.annotations[index]
        image = self.get_image_path(dataset_type, img_id)
        if self._load_pil:
            image = Image.open(image).convert('RGB')
        if self._use_subset:
            labels = [self.subsets[idx] for idx in labels if idx in self.subsets]

        if self.transforms is not None:
            image, labels = self.transforms(image, labels)
        return image, labels

    def get_image_path(self, dataset_type, img_id):
        if dataset_type == 'imagenet':
            split = self.split
            split = 'train' if split in ['train', 'training'] else split
            split = 'val' if split in ['val', 'valid', 'validation'] else split
            return os.path.join(self.root, 'images', dataset_type, 'train', img_id)
        elif dataset_type == 'open_images_v5':
            split = self.split
            split = 'train' if split in ['train', 'training'] else split
            split = 'validation' if split in ['val', 'valid', 'validation'] else split
            return os.path.join(self.root, 'images', dataset_type, split, img_id[:2], img_id[2:5], img_id + '.jpg')
        else:
            raise ValueError('not supported dataset_type:%s' % dataset_type)

    def _load_annotations(self):
        annotations = []

        # imagenet
        split = self.split
        split = 'train' if split in ['train', 'training'] else split
        split = 'val' if split in ['val', 'valid', 'validation'] else split

        file = os.path.join(self.root, 'metadata', split + '_image_id_from_imagenet_1k.txt')
        with open(file, 'r') as f:
            lines = [[l.strip() for l in line.strip().split('\t')] for line in f.readlines() if line.strip()]
            for line in lines:
                img_id, labels = line[0], [int(t.split(':')[0]) for t in line[1:]]
                annotations.append(('imagenet', img_id, labels))

        # openimages
        file = os.path.join(self.root, 'metadata', split + '_ids_from_openimages.txt')
        with open(file, 'r') as f:
            lines = [[l.strip() for l in line.strip().split('\t')] for line in f.readlines() if line.strip()]
            for line in lines:
                img_id, img_url, labels = line[0], line[1], [int(t.split(':')[0]) for t in line[2:]]
                annotations.append(('open_images_v5', img_id, labels))

        return annotations

    @staticmethod
    def _load_classes(root, lang, use_subset):
        if lang and lang != 'en':
            file = f'dictionary_and_semantic_hierarchy-{lang}.txt'
        else:
            file = f'dictionary_and_semantic_hierarchy.txt'
        file = os.path.join(root, 'metadata', file)
        classes, synsets, hierarchy = [], [], []
        with open(file, encoding='utf-8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()[1:]]
            for idx, synset_id, parent_idx, tags in lines:
                idx, parent_idx, tags = int(idx), int(parent_idx), [tag.strip() for tag in tags.split(',')]
                classes.append(tags)
                synsets.append(synset_id)
                hierarchy.append(parent_idx)

        file = os.path.join(root, 'metadata', 'tencent_ml_7m_subsets.txt')
        with open(file) as f:
            subsets = {int(idx):to_idx for to_idx, idx in enumerate(f.read().strip().split(','))}

        if use_subset:
            classes = [classes[idx] for idx in subsets.keys()]
            synsets = [synsets[idx] for idx in subsets.keys()]
            hierarchy = [subsets[hierarchy[idx]] if hierarchy[idx] >= 0 else hierarchy[idx] for idx in subsets]

        return classes, synsets, hierarchy, subsets

    @classmethod
    def get_classes(cls, lang='en', use_subset=False):
        return cls._load_classes(cls.root, lang, use_subset)
