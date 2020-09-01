import os
import csv
from collections import namedtuple, OrderedDict, defaultdict
from functools import lru_cache
from typing import List

from torchvision.datasets import VisionDataset
from PIL import Image

from .utils import TargetToVector


class ImageLevelLabel(VisionDataset):
    """Open Images Dataset V5 - Subset with Image-Level Labels (8,658 classes)

    Args:
        phase (str): One of (train, validation, test)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    Attributes:
        classes: OrderedDict[mid, label]
    """
    root = '/data/opensets/open_images_v5'
    phases = {'train', 'validation', 'test'}
    _Record = namedtuple('_Record', ['image_id', 'source', 'label_name', 'confidence'])
    classes: 'OrderedDict[str, str]'
    categories = None

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None, load_pil=True):
        super().__init__(root=self.root,
                         transforms=transforms, transform=transform, target_transform=target_transform)
        assert split in self.phases, f'{split} is not in {self.phases}'
        self.split = split
        self._load_pil = load_pil
        self._load_annotation()
        TargetToVector.inject_classes(self.transforms, self.classes)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, (positive_labels, negative_labels))
        """
        img_id: str
        anns: List[ImageLevelLabel._Record]
        img_id, (positive_ids, negative_ids) = self.annotations[index]
        image = self.get_image_path(img_id)
        if self._load_pil:
            image = Image.open(image).convert('RGB')
        label = positive_ids, negative_ids
        if self.transforms is not None:
            image, label = self.transforms(image, label)
        return image, label

    def get_image_path(self, img_id):
        return os.path.join(self.root, self.split, img_id[:2], img_id[2:5], img_id + '.jpg')

    def _load_annotation(self):
        self.classes = self._load_classes(self.root)
        self.categories = tuple(self.classes.values())

        file = os.path.join(self.root, 'metadata', 'downloaded_image_ids.csv')
        if os.path.exists(file):
            # to skip os.path.exists (it's too slow) using metadata/downloaded_image_ids.csv file
            image_ids = set()
            with open(file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header == ['ImageID', 'Subset']
                for row in reader:
                    if row[1] == self.split:
                        image_ids.add(row[0])
            is_exists = image_ids.__contains__
        else:
            @lru_cache(maxsize=128)
            def is_exists(img_id):
                return os.path.exists(self.get_image_path(img_id))

        file = os.path.join(self.root, 'annotations', '{}-annotations-human-imagelabels.csv'.format(self.split))
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            annotations = defaultdict(lambda: ([], []))
            for row in reader:
                record = ImageLevelLabel._Record._make(row)
                if record.label_name in self.classes and is_exists(record.image_id):
                    if record.confidence == '1':
                        annotations[record.image_id][0].append(record.label_name)
                    elif record.confidence == '0':
                        annotations[record.image_id][1].append(record.label_name)
            self.annotations = list(annotations.items())

    @staticmethod
    def _load_classes(root, lang='ko'):
        if lang and lang != 'en':
            file = f'class-descriptions-{lang}.csv'
        else:
            file = f'class-descriptions.csv'
        file = os.path.join(root, 'metadata', file)
        with open(file, encoding='utf-8') as f:
            reader = csv.reader(f)
            classes = dict(reader)
        file = os.path.join(root, 'metadata', 'classes-trainable.txt')
        with open(file, encoding='utf-8') as f:
            trainable_classes = f.read().strip().split()
        classes = OrderedDict((mid, classes[mid]) for mid in trainable_classes)
        return classes

    @classmethod
    def get_categories(cls, lang='ko'):
        classes = cls._load_classes(cls.root, lang=lang)
        categories = tuple(classes.values())
        return categories

    @classmethod
    def get_classes(cls, lang=None):
        return cls._load_classes(cls.root, lang=lang)
