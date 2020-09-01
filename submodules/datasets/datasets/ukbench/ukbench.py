import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader
import glob


class UKBench(VisionDataset):
    """
    Dataset UKBench
    Number of datapoints: 10200
    Root location: /data/public/ro/dataset/ukbench

    Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        load_pil (bool): load PIL.Image or path.
    
    Attributes:
        clusters (list): (cluster_id, images)    
    """

    root = '/data/public/ro/dataset/ukbench'

    def __init__(self, transforms=None, transform=None, target_transform=None, 
                 root=None, load_pil=True):
        root = root or self.root
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._load_pil = load_pil
        self.load_data()

    def load_data(self):
        pattern = os.path.join(self.root, 'full', '*.jpg')
        images = sorted(glob.glob(pattern))
        clusters = [images[i:i+4] for i in range(0, len(images), 4)]
        data = []
        for i, cluster in enumerate(clusters):
            data.extend([(c, i) for c in cluster])
        self.data = data
        self.clusters = clusters

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index]
        if self._load_pil:
            img = pil_loader(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
