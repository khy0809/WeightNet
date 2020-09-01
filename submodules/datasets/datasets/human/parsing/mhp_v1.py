from pathlib import Path
from torchvision.datasets import VisionDataset
import numpy as np
from PIL import Image


class MHPv1(VisionDataset):
    """
    MHP dataset : Multi-Human Parsing
    V1은 human parsing 만 있고, v2는 pose 포함

    https://github.com/ZhaoJ9014/Multi-Human-Parsing
    or https://lv-mhp.github.io/

    The MHP v1.0 dataset contains 4,980 images,
    each with at least two persons (average is 3).
    We randomly choose 980 images and their corresponding annotations as the testing set.
    The rest form a training set of 3,000 images and a validation set of 1,000 images.
    For each instance, 18 semantic categories are defined and annotated except for the
    "background" category, i.e. “hat”, “hair”, “sunglasses”, “upper clothes”, “skirt”,
    “pants”, “dress”, “belt”, “left shoe”, “right shoe”, “face”, “left leg”, “right leg”,
    “left arm”, “right arm”, “bag”, “scarf” and “torso skin”.
    Each instance has a complete set of annotations whenever the corresponding category
    appears in the current image.

    List of contents:

    ./images:
        All images in the dataset.

    ./annotations
        The segmentation annotation files corresponding to the images.

        One image is corresponding to multiple annotation files with the same prefix, one file per person. In each annotation file, the label represents:

        0:  'background',
        1:  'hat',
        2:  'hair',
        3:  'sunglass',
        4:  'upper-clothes',
        5:  'skirt',
        6:  'pants',
        7:  'dress',
        8:  'belt',
        9:  'left-shoe',
        10: 'right-shoe',
        11: 'face',
        12: 'left-leg',
        13: 'right-leg',
        14: 'left-arm',
        15: 'right-arm',
        16: 'bag',
        17: 'scarf',
        18: 'torso-skin',

    ./visualization.m
        Matlab script to visualize the annotations


    ./train_list.txt  4000개
        The list of images for training and validataion

    ./test_list.txt 980개
        The list of images for testing
    """
    root = '/data/public/rw/datasets/human/parsing/LV-MHP-v1'
    category = ('__background__', 'hat', 'hair', 'sunglass', 'upper-clothes',
                'skirt', 'pants', 'dress', 'belt', 'left-shoe',
                'right-shoe', 'face', 'left-leg', 'right-leg', 'left-arm',
                'right-arm', 'bag', 'scarf', 'torso-skin',)

    def __init__(self, what='train', transforms=None, transform=None, target_transform=None, root=None):
        root = root or MHPv1.root
        super(MHPv1, self).__init__(root=root, transforms=transforms,
                                    transform=transform, target_transform=target_transform)
        assert what in ('train', 'test')
        self.what = what
        root = Path(root)
        self.imagepath = root / 'images'
        self.annopath = root / 'annotations'
        fname = root / f'{what}_list.txt'
        with open(fname, 'r') as f:
            image_ids = [line.split('.jpg')[0] for line in f.readlines()]
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        i = self.image_ids[index]
        fname = self.imagepath / f'{i}.jpg'
        image = Image.open(fname)
        files = self.annopath.glob(f'{i}_*.png')
        anno = [Image.open(f) for f in files]

        return image, anno

