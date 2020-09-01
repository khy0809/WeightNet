import os
import csv
from torchvision.datasets import VisionDataset
from PIL import Image


class Aptos2019(VisionDataset):
    """
    APTOS 2019 Blindness Detection

    You are provided with a large set of retina images taken using fundus photography under a variety of imaging conditions.

    A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4

    https://www.kaggle.com/c/aptos2019-blindness-detection
    """

    root = '/data/public/ro/dataset/images/aptos2019'

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None):
        """

        Args:
            split: One of (train, test)
        """
        super(Aptos2019, self).__init__(self.root,
                                        transforms=transforms, transform=transform, target_transform=target_transform)

        if split == 'train':
            csv_path = os.path.join(self.root, 'train.csv')
            self.img_root = os.path.join(self.root, 'train')
        elif split == 'test':
            csv_path = os.path.join(self.root, 'test.csv')
            self.img_root = os.path.join(self.root, 'test')
        else:
            raise NotImplementedError(f'Unexpected split: {split}')

        csv_reader = csv.reader(open(csv_path, 'r'))
        imgs = [row for row in csv_reader][1:]

        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        if len(self.imgs[index]) == 2:
            imgid, label = self.imgs[index]
            label = int(label)
        else:
            imgid, = self.imgs[index]
            label = None

        image = Image.open(os.path.join(self.img_root, f'{imgid}.png'))

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label


if __name__ == '__main__':
    for what in ['train', 'test']:
        dataset = Aptos2019(split=what)
        print(f'len({what}) = {len(dataset)}')
        for i, (image, target) in enumerate(dataset):
            print(i, image.size, target)

            image.save(f'{what}_{i}.jpg')
            if i == 10:
                break
