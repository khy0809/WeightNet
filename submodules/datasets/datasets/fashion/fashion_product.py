import os
from torchvision.datasets import VisionDataset
from PIL import Image
import datasets.util.json as json

# label/ tag 구조가 small 데이터셋과 같음
from .fashion_product_small import (keys, productinfo, _read_styles_csv)


class FashionProduct(VisionDataset):
    """fashion-product dataset
    from https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset
    Context:
        The growing e-commerce industry presents us with a large dataset waiting to be scraped and researched upon.
        In addition to professionally shot high resolution product images,
        we also have multiple label attributes describing the product
        which was manually entered while cataloging.
        To add to this, we also have descriptive text that comments on the product characteristics.

    Content:
        Each product is identified by an ID like 42431. You will find a map to all the products in styles.csv.
        From here, you can fetch the image for this product from images/42431.jpg and
        the complete metadata from styles/42431.json.

        To get started easily, we also have exposed some of the key product categories and
        it's display name in styles.csv.

        If this dataset is too large, you can start with a smaller (280MB) version
        here: https://www.kaggle.com/paramaggarwal/fashion-product-images-small

    Inspiration:
        So what can you try building? Here are some suggestions:

        Start with an image classifier. Use the masterCategory column from styles.csv and
        train a convolutional neural network.
        The same can be achieved via NLP.
        Extract the product descriptions from styles/42431.json and
        then run a classifier to get the masterCategory.
        Try adding more sophisticated classification by predicting the other category labels in styles.csv
        Transfer Learning is your friend and use it wisely.
        You can even take things much further from here:

        Is it possible to build a GAN that takes a category as input and outputs an image?
        Auto-encode the image attributes to be able to make a visual search engine
        that converts the image into a small encoding which is sent to the server to perform visual search?
        Visual similarity search? Given an image, suggest other similar images.
    """

    root = '/data/public/rw/datasets/fashion/fashion-product'
    image_path = f'{root}/images'
    json_path = f'{root}/styles'

    def __init__(self, train=None, transforms=None, transform=None, target_transform=None):
        super().__init__(root=self.root, transforms=transforms,
                         transform=transform, target_transform=target_transform)
        if train is not None:
            raise NotImplementedError('splits not available')
        self.products = _read_styles_csv(self.root)

    def __len__(self):
        return len(self.products)

    def __getitem__(self, index):
        label = self.products[index]
        file = self._imagepath(label.id)
        # assert os.path.exists(file)
        meta = self._jsontpath(label.id)
        img = Image.open(file)
        meta = json.load(open(meta, 'r'))

        if self.transforms:
            img, (label, meta) = self.transforms(img, meta)

        return img, (label, meta)

    def _imagepath(self, id):
        return f'{self.image_path}/{id}.jpg'

    def _jsontpath(self, id):
        return f'{self.json_path}/{id}.json'


def _read_images_csv(csv):
    with open(csv, 'r') as f:
        f.readline()  # pass first header
        for line in f.readlines():
            img, url = line.strip().split(',')
            yield img, url
