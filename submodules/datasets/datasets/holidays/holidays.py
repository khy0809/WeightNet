import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader
import glob
from collections import defaultdict


class Holidays(VisionDataset):
    """Holidays dataset
    http://lear.inrialpes.fr/people/jegou/data.php#holidays

    Dataset Holidays
        Number of datapoints: 1491
        Root location: /data/public/ro/dataset/holidays

    Args:
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
        clusters (list): (cluster_id, images)

    """
    root = '/data/public/ro/dataset/holidays'

    def __init__(self, transforms=None, transform=None, target_transform=None, 
                 root=None, load_pil=True):
        root = root or self.root
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._load_pil = load_pil
        self.imgpath = os.path.join(root, 'jpg')
        self.load_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index]
        if self._load_pil:
            img = pil_loader(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def load_images(self):
        pattern = os.path.join(self.imgpath, '*.jpg')
        images = sorted(glob.glob(pattern))
        
        grouped = defaultdict(list)
        for path in images:
            image_id = int(os.path.basename(path[:-len('.jpg')]))
            grouped[image_id // 100].append(path)
        
        self.clusters = list(grouped.values())
        data = []
        for i, paths in enumerate(self.clusters):
            for p in paths:
                data.append((p, i))
        self.data = data
