import os
from torchvision.datasets import VisionDataset
from PIL import Image
from .places365 import Places365
from .places_extra69 import PlacesExtra69


class Places434(VisionDataset):
    """Places434
    Places365 + PlacesExtra69
    
    Args:
        split (str): One of (train, test)
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

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None,
                 large=True, challenge=False, root=None, load_pil=True):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        assert split in self.splits, f'{split} is not in {self.splits}'
        self.split = split
        self.large = large
        self.challenge = challenge
        options = dict(large=large, root=root, load_pil=load_pil)
        self.p365 = Places365(split=split, challenge=challenge, **options)
        self.n365 = len(self.p365)
        self.ex69 = PlacesExtra69(split=split, **options)
        self.classes = self.get_categories()

    def __len__(self):
        return len(self.p365) + len(self.ex69)

    def __getitem__(self, index):
        if index < 0:
            index %= len(self)
        if 0 <= index < self.n365:
            img, target = self.p365[index]
        else:
            img, target = self.ex69[index - self.n365]
            target = None if target is None else target + 365
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __repr__(self):
        r = super().__repr__()
        r += f"\n    image size: {'large' if self.large else '256'}"
        r += f"\n    type: {'challenge' if self.challenge else 'standard'}"
        r += f"\n    split: {self.split}"
        r += f"\n    # of categories: {len(self.classes)}"
        return r

    @classmethod
    def get_categories(cls):
        """
        get_categories: label strings
        """
        cate = list(Places365.get_categories())
        cate += list(PlacesExtra69.get_categories())

        return tuple(cate)
