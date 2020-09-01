from __future__ import print_function
import os
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset


class CINIC10(VisionDataset):
    """
    CINIC-10: CINIC-10 Is Not Imagenet or CIFAR-10
    """
    root = '/data/opensets/cinic10'
    splits = {'train', 'val', 'test'}

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None):
        assert split in self.splits, f'{split} is not in {self.splits}'
        super().__init__(root=CINIC10.root, transforms=transforms, transform=transform, target_transform=target_transform)

        if split == 'val':
            split = 'valid'

        df = pd.read_csv(os.path.join(CINIC10.root, f'{split}.csv'))

        self.split = split
        self.cls_map = list(df['class'].unique())
        self.imglist = df

    def __getitem__(self, index):
        row = self.imglist.iloc[index]
        image = Image.open(os.path.join(CINIC10.root, self.split, row['class'], row['img_file']))
        label = self.cls_map.index(row['class'])

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self):
        return len(self.imglist)


if __name__ == '__main__':
    for what in ['train', 'val', 'test']:
        dataset = CINIC10(split=what)
        print(f'len({what}) = {len(dataset)}')
        for i, (image, target) in enumerate(dataset):
            print(i, image.size, target)

            image.save(f'/root/{what}_{i}.jpg')
            if i == 10:
                break
