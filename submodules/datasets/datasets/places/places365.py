import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader


def _read_categories_file(catefile):
    categories = dict()
    with open(catefile) as f:
        for line in f:
            splits = line.strip().split()
            categories[int(splits[1])] = splits[0]
    return categories


def _read_image_list(file):
    with open(file) as f:
        for line in f:
            splits = line.strip().split()
            yield splits


def _read_scene_hierarchy(f):
    hierachies = dict()
    with open(f) as f:
        # level1: 3 type
        # level2-indoor: 6 type
        # level2-outdoor natural: 4 type
        # level2-outdoor man-made: 6 type
        next(f) # skip first line
        descriptions = next(f).strip().split('\t')[1:]
        for line in f:
            splits = line.strip().split('\t')
            category = splits[0][3:]
            flags = [bool(int(i)) for i in splits[1:]]
            hierachies[category] = flags
    return hierachies, descriptions


class Places365(VisionDataset):
    """Places365

    Args:
        split (str): One of (train, val, test)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        large (bool): load large one or 256 x 256 image
        challenge (bool): if True, load the challenge dataset else standard
        load_pil (bool): load PIL.Image or path.
    Attributes:
        classes: labels of classes
    """

    root = '/data/opensets/places'
    splits = ('train', 'val', 'test')
    classes = None
    categories = None

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None,
                 large=True, challenge=False, root=None, load_pil=True):
        root = root or self.root
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        assert split in self.splits, f'{split} is not in {self.splits}'
        self.split = split
        imgpath = dict(train='data', test='test', val='val')[split]
        self.large = large
        
        # set image path
        imgpath = f"{imgpath}_{'large' if large else '256'}"
        self.imgpath = os.path.join(root, imgpath)
        self.challenge = challenge

        # read the image list
        if split == 'train':
            file = f"places365_train_{'challenge' if challenge else 'standard'}.txt"
        else:
            file = f"places365_{split}.txt"
        file = os.path.join(root, file)
        self.data = tuple(splits for splits in _read_image_list(file))
        # read catgory info
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
        r += f"\n    type: {'challenge' if self.challenge else 'standard'}"
        r += f"\n    split: {self.split}"
        r += f"\n    # of categories: {len(self.classes)}"
        return r

    @classmethod
    def get_categories(cls):
        """
        get_categories: label strings
        """
        catefile = os.path.join(cls.root, 'categories_places365.txt')
        cate = _read_categories_file(catefile)
        # ignore first 3 characters. (ex: `/a/` in `/a/airfield`)
        return tuple(cate[i][3:] for i in range(365))

    @classmethod
    def get_scene_hierachy(cls):
        """get scene hierachical information
            flags
            level1: 3 type (indoor, outdoor natural, outdoor man-made)
            level2-indoor: 6 type
            level2-outdoor natural: 4 type
            level2-outdoor man-made: 6 type
        Returns:
            hierachy (dict): Dict[category, flags]
            desc list[str]: descriptions for flags len(desc) == 19
        """
        f = os.path.join(cls.root, 'places365_scene_hierachy.tsv')
        hierachy, desc = _read_scene_hierarchy(f)
        return hierachy, desc
