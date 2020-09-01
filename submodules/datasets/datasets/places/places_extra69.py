import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader

from .places365 import (_read_categories_file, _read_image_list)


class PlacesExtra69(VisionDataset):
    """PlacesExtra69

    Args:
        split (str): One of (train, val, test)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        large (bool): load large one or 256 x 256 image
        load_pil (bool): load PIL.Image or path.
    Attributes:
        classes: labels of classes
    """
    root = '/data/opensets/places'
    splits = ('train', 'test')
    classes = None
    categories = None

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None,
                large=True, root=None, load_pil=True):
        root = root or self.root
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        assert split in self.splits, f'{split} is not in {self.splits}'
        self.split = split
        imgpath = dict(train='data', test='test', val='val')[split]
        self.large = large
        
        if large:
            imgpath = 'data_large_extra'
            file = f"extra69_large_{split}.txt"
        else:
            imgpath = 'data_256_extra'
            file = f"extra69_{split}.txt"
        self.imgpath = os.path.join(root, imgpath)

        file = os.path.join(root, file)
        self.data = tuple(splits for splits in _read_image_list(file))
        self.classes = self.get_categories()
        self._load_pil = load_pil

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        img = self.get_image_path(d[0])
        if self._load_pil:
            img = pil_loader(img)
        try:
            target = int(d[1])
        except IndexError:
            target = None
        if self.transforms is not None:
            return self.transforms(img, target)
        return img, target

    def get_image_path(self, f):
        if f.startswith('/'):
            return os.path.join(self.imgpath, f[1:])
        else:
            return os.path.join(self.imgpath, f)

    def __repr__(self):
        r = super().__repr__()
        r += f"\n    image size: {'large' if self.large else '256'}"
        r += f"\n    split: {self.split}"
        r += f"\n    # of categories: {len(self.classes)}"
        return r

    @classmethod
    def get_categories(cls):
        """
        get_categories: label strings
        """
        catefile = os.path.join(cls.root, 'categories_extra69.txt')
        cate = _read_categories_file(catefile)
        # ignore first 3 characters. (ex: `/a/` in `/a/airfield`)
        return tuple(cate[i][3:] for i in range(69))
