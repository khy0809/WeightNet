from torchvision.datasets import VisionDataset
from PIL import Image
import os
import pickle


class KakaoStyleInfo(VisionDataset):

    root = '/data/project/rw/kakaostyle/'

    def __init__(self, transforms=None, transform=None, target_transform=None):
        super().__init__(root=self.root, transforms=transforms,
                         transform=transform, target_transform=target_transform)
        f = os.path.join(self.root, 'itemid_url.pkl')
        self.itemurls = pickle.load(open(f, 'rb'))

        f = os.path.join(self.root, 'iteminfo.pkl')
        self.iteminfo = pickle.load(open(f, 'rb'))

    def __len__(self):
        return len(self.itemurls)

    def __getitem__(self, i):
        itemid, url = self.itemurls[i]
        category, display, tags = self.iteminfo[itemid]
        target = self.iteminfo[itemid]
        imfile = self._get_image_path(url)
        image = Image.open(imfile).convert('RGB')
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def _get_image_path(self, url):
        """tenth url path를 -> nas 폴더 path로"""
        if url.startswith('/'):
            url = url[1:]
        return os.path.join(self.root, 'image', url)
