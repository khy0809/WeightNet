import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader


class Food101(VisionDataset):
    """
    trainset, testset 두종류 합이 101 종류, 101,000 이미지. only one label
    - 클래스당 train : test = 750장: 250장
    대략 width 또는 height 한쪽은 512 크기로 보입니다. 혹은 512 x 512
    ├── images
    │   ├── apple_pie
    │   ├── baby_back_ribs
    │   ├── baklava
    │   ...
    │   └── waffles
    └── meta
    """

    # tf-talkdrawer bc-workspace
    root = '/data/opensets/food101'
    labels = None  # init 에서 채워진다.

    def __init__(self, train=True, transforms=None, transform=None, target_transform=None, root=None):
        root = root or self.root
        self.what = 'train' if train else 'test'
        super(Food101, self).__init__(root, transforms=transforms, transform=transform,
                                      target_transform=target_transform)
        assert os.path.exists(root), f'root folder [{root}] not found. check your bc-project'
        self.image_dir = os.path.join(root, 'images')
        meta_dir = os.path.join(root, 'meta')

        # 순서가 중요합니다.
        with open(os.path.join(meta_dir, 'classes.txt'), 'r') as f:
            self.classes = [line.strip() for line in f]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        with open(os.path.join(meta_dir, 'labels.txt'), 'r') as f:
            self.labels = [line.strip() for line in f]

        # meta/train.txt or meta/test.txt
        with open(os.path.join(meta_dir, f'{self.what}.txt'), 'r') as f:
            self._files = [line.strip() for line in f]

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        """
        :param i: 0 ~ len(self)
        :return: PIL.Image, the index of class
        """
        # ex) f = 'apple_pie/1026328'
        f = self._files[i]
        folder, index = f.split('/')
        icate = self.class_to_idx[folder]
        img_path = os.path.join(self.image_dir, f'{f}.jpg')
        image = pil_loader(img_path)
        label = icate
        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label
