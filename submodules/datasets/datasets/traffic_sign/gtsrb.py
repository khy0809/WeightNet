import os
import csv
import glob
from torchvision.datasets import VisionDataset
from PIL import Image


class GTSRB(VisionDataset):
    """
    German Traffic Sign Recognition Benchmark GTSRB

    The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class image classification benchmark in the domain of advanced driver assistance systems and autonomous driving.

    http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
    """

    root = '/data/opensets/GTSRB'

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None):
        """

        Args:
            split: One of (train, test)
        """
        super(GTSRB, self).__init__(self.root,
                                    transforms=transforms, transform=transform, target_transform=target_transform)

        if split == 'train':
            csvs = glob.glob(f'{self.root}/Final_Training/Images/*/GT-*.csv')
            self.img_root = '{}/Final_Training/Images/{:05d}'
        elif split == 'test':
            csvs = [f'{self.root}/GT-final_test.csv']
            self.img_root = '{}/Final_Test/Images'
        else:
            raise NotImplementedError(f'Unexpected split: {split}')

        imgs = []
        for c in csvs:
            csv_reader = csv.reader(open(c, 'r'), delimiter=';')
            imgs += [row for row in csv_reader][1:]

        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        filename, width, height, x1, y1, x2, y2, cls = self.imgs[index]

        x1, y1, x2, y2, cls = map(int, (x1, y1, x2, y2, cls))

        image = Image.open(os.path.join(self.img_root.format(self.root, cls), filename))

        # this dataset comes with RoI for the classification task. RoI seems to be needed as GT for the detection task
        image = image.crop((x1, y1, x2, y2))

        label = cls

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label


if __name__ == '__main__':
    for what in ['train', 'test']:
        dataset = GTSRB(split=what)
        print(f'len({what}) = {len(dataset)}')
        for i, (image, target) in enumerate(dataset):
            print(i, image.size, target)

            image.save(f'{what}_{i}.jpg')
            if i == 20:
                break