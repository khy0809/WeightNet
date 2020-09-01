from __future__ import print_function
import os
import shutil
import torch


import torchvision
from torchvision.datasets.utils import check_integrity, download_url


# copy ILSVRC/ImageSets/CLS-LOC/train_cls.txt to ./root/
# to skip os walk (it's too slow) using ILSVRC/ImageSets/CLS-LOC/train_cls.txt file
class ImageNet(torchvision.datasets.VisionDataset):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    ROOT = '/data/public/rw/datasets/imagenet-pytorch'
    splits = {'train', 'val'}
    categories = None

    def __init__(self, split='train', **kwargs):
        super(ImageNet, self).__init__(ImageNet.ROOT, **kwargs)

        self.split = self._verify_split(split)
        wnid_to_classes = self._load_meta_file(self.root)[0]

        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        # to skip os walk (it's too slow) using ILSVRC/ImageSets/CLS-LOC/train_cls.txt file
        listfile = os.path.join(self.root, 'train_cls.txt')
        if split == 'train' and os.path.exists(listfile):
            with open(listfile, 'r') as f:
                datalist = [
                    line.strip().split(' ')[0]
                    for line in f.readlines()
                    if line.strip()
                ]

            classes = list(set([line.split('/')[0] for line in datalist]))
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}

            samples = [
                (os.path.join(self.split_folder, line + '.JPEG'),
                 class_to_idx[line.split('/')[0]])
                for line in datalist
            ]
        else:
            classes, class_to_idx = self._find_classes(self.split_folder)
            samples = torchvision.datasets.folder.make_dataset(self.split_folder, class_to_idx, self.extensions, None)

        self.wnids = classes
        self.wnid_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        self.categories = classes

    def __getitem__(self, index, with_transform=True):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None and with_transform:
            sample, target = self.transforms(sample, target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    @property
    def meta_file(self):
        return os.path.join(self.root, 'meta.bin')

    @staticmethod
    def _load_meta_file(root):
        meta_file = os.path.join(root, 'meta.bin')
        if check_integrity(meta_file):
            return torch.load(meta_file)
        raise RuntimeError("Meta file not found or corrupted.",
                           "You can use download=True to create it.")

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val'

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    @staticmethod
    def _find_classes(dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

    @classmethod
    def get_categories(cls):
        wnid_to_classes = cls._load_meta_file(cls.ROOT)[0]
        wnids = list(sorted(wnid_to_classes.keys()))
        categories = tuple(wnid_to_classes[w] for w in wnids)

        return categories
