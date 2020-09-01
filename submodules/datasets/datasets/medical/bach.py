import os
import csv
import glob
from torchvision.datasets import VisionDataset
from PIL import Image


class BACH(VisionDataset):
    """
    ICIAR 2018 Grand Challenge on BreAst Cancer Histology images

    automatically classifying H&E stained breast histology microscopy images in four classes:
    normal, benign, in situ carcinoma and invasive carcinoma

    https://iciar2018-challenge.grand-challenge.org/Home/
    """

    root = '/data/public/ro/dataset/images/ICIAR2018_BACH_Challenge'
    cls_name_to_idx = {'Normal': 0, 'Benign': 1, 'InSitu': 2, 'Invasive': 3}

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None):
        """

        Args:
            split: One of (train, test)
        """
        super(BACH, self).__init__(self.root,
                                   transforms=transforms, transform=transform, target_transform=target_transform)

        if split == 'train':
            self.img_root = os.path.join(self.root, 'train', 'Photos')
            csv_path = os.path.join(self.img_root, 'microscopy_ground_truth.csv')
            csv_reader = csv.reader(open(csv_path, 'r'))
            self.imgs = [row for row in csv_reader]

        elif split == 'test':
            self.img_root = os.path.join(self.root, 'test', 'Photos')
            self.imgs = glob.glob(f'{self.img_root}/*.tif')

        else:
            raise NotImplementedError(f'Unexpected split: {split}')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if isinstance(self.imgs[index], list):
            img_path, label = self.imgs[index]
            img_path = os.path.join(self.img_root, label, img_path)
            label = self.cls_name_to_idx[label]
        else:
            img_path = self.imgs[index]
            label = None

        image = Image.open(img_path)

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label


if __name__ == '__main__':
    for what in ['train', 'test']:
        dataset = BACH(split=what)
        print(f'len({what}) = {len(dataset)}')
        for i, (image, target) in enumerate(dataset):
            print(i, image.size, target)

            image.save(f'/root/{what}_{i}.jpg')
            if i == 10:
                break
