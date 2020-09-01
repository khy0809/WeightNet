import os
import os.path as osp
from torchvision.datasets import VisionDataset
from PIL import Image


class RESISC45(VisionDataset):
    '''
    http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html
    '''

    root = '/data/public/rw/datasets/aerial_inspection/NWPU-RESISC45'

    def __init__(self, transforms=None, transform=None, target_transform=None):
        super().__init__(root=self.root, transforms=transforms,
                         transform=transform, target_transform=target_transform)

        self.classes = sorted(os.listdir(self.root))
        self._files = list()
        self.labels = list()

        for cls_nm in self.classes:
            class_img_root = osp.join(self.root, cls_nm)
            cls_files = sorted([osp.join(class_img_root, img)
                                for img in os.listdir(class_img_root)
                                if img.endswith('jpg')])
            self._files += cls_files
            self.labels += [self.classes.index(cls_nm)] * len(cls_files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        imfile = self._files[i]
        image = Image.open(imfile).convert('RGB')
        target = self.labels[i]

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
