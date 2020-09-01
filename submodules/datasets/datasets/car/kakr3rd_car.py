import os
import random

import pandas as pd
from PIL import Image

from torchvision.datasets import VisionDataset


class CarDataSet(VisionDataset):
    """
    2019 3rd ML month with KaKR

    자동차 이미지 데이터셋을 이용한 자동차 차종 분류
    자동차 데이터 셋은 9,990개의 Train 셋, 6,150개의 Test 셋 을 합하여 총 16,140개의 이미지와 바운딩박스 좌표, 차종정보로 이루어져 있습니다.

    https://www.kaggle.com/c/2019-3rd-ml-month-with-kakr
    """
    root = '/data/public/ro/dataset/images/car-kakr-3rd-mlmonth'

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None):
        """

        Args:
            split: One of (train, test)
        """
        assert split in ['train', 'test']
        super(CarDataSet, self).__init__(self.root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.annos = pd.read_csv(os.path.join(self.root, f'{split}.csv'))
        self.split = split
        self.crop_ratio = (0.9, 1.1) if split == 'train' else (1.0, 1.0)
        self.crop_aug = 0.1 if split == 'train' else 0.0

    def __getitem__(self, index):
        filepath = self.annos['img_file'][index]
        x1 = self.annos['bbox_x1'][index]
        x2 = self.annos['bbox_x2'][index]
        y1 = self.annos['bbox_y1'][index]
        y2 = self.annos['bbox_y2'][index]
        lb = self.annos['class'][index] - 1 if 'class' in self.annos else 0
        image = Image.open(os.path.join(self.root, self.split, filepath))

        # crop image
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if isinstance(self.crop_ratio, tuple):
            crop_ratio_w = random.uniform(self.crop_ratio[0], self.crop_ratio[1])
            crop_ratio_h = random.uniform(self.crop_ratio[0], self.crop_ratio[1])
            w, h = int(crop_ratio_w * (x2 - x1)), int(crop_ratio_h * (y2 - y1))
        else:
            w, h = int(self.crop_ratio * (x2 - x1)), int(self.crop_ratio * (y2 - y1))

        center_x += int(w * random.uniform(-self.crop_aug, self.crop_aug))
        center_y += int(h * random.uniform(-self.crop_aug, self.crop_aug))

        x1 = center_x - w // 2
        x2 = x1 + w
        y1 = center_y - h // 2
        y2 = y1 + h

        real_w = min(image.width, x2) - max(0, x1)
        real_h = min(image.height, y2) - max(0, y1)

        bg = Image.new('RGB', (x2 - x1, y2 - y1), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        cropped = image.crop((max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)))
        bg.paste(cropped, (random.randint(0, w - real_w), random.randint(0, h - real_h)))
        image = bg

        if self.transforms is not None:
            image, lb = self.transforms(image, lb)

        return image, lb

    def __len__(self):
        return len(self.annos)


if __name__ == '__main__':
    d = CarDataSet()
    img, lb = d[0]
    print(img)
    print(lb)
