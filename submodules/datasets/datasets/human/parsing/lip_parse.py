import os
from torchvision.datasets import VisionDataset
from PIL import Image


class LIPparse(VisionDataset):
    """Look Into Person for human parsing
    see : http://sysu-hcp.net/lip/overview.php
    human parsing 데이터셋 클래스 (single person) - nas 버전

    파싱 target은 PNG 파일 (channel 없는 값)
    Target Labels:
        0.Background
        1.Hat
        2.Hair
        3.Glove
        4.Sunglasses
        5.UpperClothes
        6.Dress
        7.Coat
        8.Socks
        9.Pants
        10.Jumpsuits
        11.Scarf
        12.Skirt
        13.Face
        14.Left-arm
        15.Right-arm
        16.Left-leg
        17.Right-leg
        18.Left-shoe
        19.Right-shoe
    """
    root = '/data/public/rw/datasets/human/LIP'
    whats = ('train', 'val', 'test')
    whatfolders = ('train_images', 'val_images', 'testing_images')
    labels = ('Background', 'Hat', 'Hair', 'Glove', 'Sunglasses',
              'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
              'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
              'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe')

    def __init__(self, what='train', transforms=None, transform=None, target_transform=None):
        super().__init__(root=LIPparse.root,
                         transforms=transforms, transform=transform, target_transform=target_transform)
        assert what in self.whats, f'{what} is not in {self.whats}'
        self.what = what
        self._whatfolder = self.whatfolders[self.whats.index(what)]
        # setup for what
        self.ids = self._load_ids()

    def _load_ids(self):
        file = os.path.join(self.root, f'{self.what}_id.txt')
        with open(file, 'rt') as f:
            return [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        f = self.ids[index]
        image = os.path.join(self.root, self._whatfolder, f'{f}.jpg')
        image = Image.open(image)

        if self.what != 'test':
            # {root}/TrainVal_parsing_annotations/train_segmentations/
            file = f'TrainVal_parsing_annotations/{self.what}_segmentations/{f}.png'
            label = os.path.join(self.root, file)
            label = Image.open(label)
        else:
            label = None

        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label

