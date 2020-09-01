import os
import csv
from collections import namedtuple, OrderedDict, defaultdict

from torchvision.datasets import VisionDataset
from PIL import Image

from .utils import TargetToVector


class ImageLevelLabelBoxable(VisionDataset):
    """Open Images Dataset V5 - Image-level Multi-labels by Box annotation and labels

    Args:
        split (str): One of (train, validation, test)
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
    splits = {'train', 'validation', 'test'}
    _Record = namedtuple('_Record', ['image_id', 'source', 'label_name', 'confidence'])
    classes: 'OrderedDict[str, str]'
    categories = None

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None, root=None,
                 load_pil=True):
        root = root or self.root
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        assert split in self.splits, f'{split} is not in {self.splits}'
        self.split = split
        self._load_pil = load_pil
        self._load_annotations()
        TargetToVector.inject_classes(self.transforms, self.classes)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id, label = self.annotations[index]
        image = self.get_image_path(img_id)
        if self._load_pil:
            image = Image.open(image).convert('RGB')
        if self.transforms is not None:
            image, label = self.transforms(image, label)
        return image, label

    def _load_annotations(self):
        self.classes = self._load_classes(self.root)
        self.categories = tuple(self.classes.values())

        # get positive label from bbox
        file = os.path.join(self.root, 'annotations', '{}-annotations-bbox.csv'.format(self.split))

        Record = ImageLevelLabelBoxable._Record
        with open(file, 'r') as reader:
            # skip header row
            # ImageID,Source,LabelName,Confidence,...
            next(reader)
            positives = defaultdict(list)
            for line in reader:
                # ImageID,Source,LabelName,Confidence
                row = line.split(',', maxsplit=3)
                record = Record(*row)
                positives[record.image_id].append(record.label_name)

        # get negative labels from image-level annotations
        file = os.path.join(self.root, 'annotations', '{}-annotations-human-imagelabels-boxable.csv'.format(self.split))
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            negatives = defaultdict(list)
            for row in reader:
                if row[-1] != '0':
                    # skip positive image-level labels
                    continue
                record = Record(*row)
                negatives[record.image_id].append(record.label_name)

        annotations = {img_id: (list(set(pos)), list(set(negatives[img_id])))
                       for img_id, pos in positives.items()}
        self.annotations = list(annotations.items())

    def get_image_path(self, img_id):
        return os.path.join(self.root, self.split, img_id[:2], img_id[2:5], img_id + '.jpg')

    @staticmethod
    def _load_classes(root, lang='ko'):
        if lang and lang != 'en':
            file = f'class-descriptions-boxable-{lang}.csv'
        else:
            file = f'class-descriptions-boxable.csv'
        file = os.path.join(root, 'metadata', file)
        with open(file, encoding='utf-8') as f:
            reader = csv.reader(f)
            classes = OrderedDict(reader)
        return classes

    @classmethod
    def get_categories(cls, lang='ko'):
        classes = cls._load_classes(cls.root, lang=lang)
        categories = tuple(classes.values())
        return categories

    @classmethod
    def get_classes(cls, lang=None):
        return cls._load_classes(cls.root, lang=lang)
