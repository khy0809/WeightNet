import os
import scipy.io
from torchvision.datasets import VisionDataset
from PIL import Image


class Flowers102(VisionDataset):
    """
    102 category dataset, consisting of 102 flower categories. Each class consists of between 40 and 258 images.

    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
    """

    root = '/data/opensets/flowers102'

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None):
        """

        Args:
            split: One of (train, validation, test)
        """
        super(Flowers102, self).__init__(self.root,
                                         transforms=transforms, transform=transform, target_transform=target_transform)

        imagelabels = scipy.io.loadmat(os.path.join(self.root, 'imagelabels.mat'))
        self.labels = imagelabels['labels'].reshape(-1)

        setid = scipy.io.loadmat(os.path.join(self.root, 'setid.mat'))

        if split == 'train':
            imgs = setid['trnid']
        elif split == 'validation':
            imgs = setid['valid']
        elif split == 'test':
            imgs = setid['tstid']
        else:
            raise NotImplementedError(f'Unexpected split: {split}')

        self.imgs = imgs.reshape(-1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        imgid = self.imgs[index]

        image = Image.open(os.path.join(self.root, 'jpg', f'image_{imgid:05d}.jpg'))
        label = self.labels[imgid - 1] # imgid is 1-based, labels array is not

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label


if __name__ == '__main__':
    for what in ['train', 'validation', 'test']:
        dataset = Flowers102(split=what)
        print(f'len({what}) = {len(dataset)}')
        for i, (image, target) in enumerate(dataset):
            print(i, image.size, target)

            image.save(f'{what}_{i}.jpg')
            if i == 10:
                break
