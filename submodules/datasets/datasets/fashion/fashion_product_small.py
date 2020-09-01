# import os
from torchvision.datasets import VisionDataset
from collections import namedtuple
from PIL import Image

keys = 'id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName'.strip().split(',')
productinfo = namedtuple('productinfo', keys)


class FashionProductSmall(VisionDataset):
    """ fashion-product-small dataset
    from https://www.kaggle.com/paramaggarwal/fashion-product-images-small/downloads/fashion-product-images-small.zip/1
    44k image data and the meta information of
        ['id', 'gender', 'masterCategory', 'subCategory', 'articleType',
        'baseColour', 'season', 'year', 'usage', 'productDisplayName']

    Context:
        The growing e-commerce industry presents us with a large dataset waiting to be scraped and researched upon.
        In addition to professionally shot high resolution product images, we also have multiple label attributes
        describing the product which was manually entered while cataloging.
        To add to this, we also have descriptive text that comments on the product characteristics.

    Content:
        Each product is identified by an ID like 42431.
        You will find a map to all the products in styles.csv.
        From here, you can fetch the image for this product from images/42431.jpg.
        To get started easily,
        we also have exposed some of the key product categories and it's display name in styles.csv.

    Inspiration:
        So what can you try building? Here are some suggestions:

        Start with an image classifier.
        Use the masterCategory column from styles.csv and train a convolutional neural network.
        The same can be achieved via NLP. Extract the product descriptions from styles/42431.json and then
        run a classifier to get the masterCategory.
        Try adding more sophisticated classification by predicting the other category labels in styles.csv
        Once you are ready to upgrade, go to the high resolution image (2400x1600)
        dataset: https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset
        try `FashionProduct` dataset
    """

    # tenth
    root = '/data/public/rw/datasets/fashion/fashion-product-small'
    image_path = f'{root}/images'

    def __init__(self, train=None, transforms=None, transform=None, target_transform=None):
        # root = '/data/public/rw/datasets/fashion/fashion-product-small'
        super().__init__(root=self.root, transforms=transforms,
                         transform=transform, target_transform=target_transform)

        if train is not None:
            raise NotImplementedError('splits not yet')

        self.products = _read_styles_csv(self.root)

    def __len__(self):
        return len(self.products)

    def __getitem__(self, index):
        label = self.products[index]
        image = Image.open(self._getpath(label.id))

        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label

    def _getpath(self, id):
        return f'{self.root}/images/{id}.jpg'


def _read_styles_csv(root):
    """ return [(id, json), ...]
    """
    path = f'{root}/styles.csv'
    products = []

    with open(path, 'r') as f:
        header = f.readline().split(',')
        for line in f.readlines():
            split = line.strip().split(',')
            # 아마 productDisplayName 에 컴마 있는 경우들
            products.append(productinfo(*split[:9], ' '.join(split[9:])))
    return products

