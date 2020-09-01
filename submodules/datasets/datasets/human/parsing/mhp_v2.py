from pathlib import Path
from torchvision.datasets import VisionDataset
import numpy as np
from PIL import Image
from collections import OrderedDict
import scipy.io


class MHPv2(VisionDataset):
    """
    MHP dataset : Multi-Human Parsing v2 (for parsing)

    https://github.com/ZhaoJ9014/Multi-Human-Parsing
    or https://lv-mhp.github.io/dataset

    The MHP v2.0 dataset contains 25,403 images, each with at least two persons (average is 3).
    We randomly choose 5,000 images and their corresponding annotations as the testing set.
    The rest form a training set of 15,403 images and a validation set of 5,000 images.
    For each instance, 58 semantic categories are defined and annotated except for the "background" category,
    i.e. “cap/hat", “helmet", “face", “hair", “left- arm", “right-arm", “left-hand", “right-hand",
    “protector", “bikini/bra", “jacket/windbreaker/hoodie", “t-shirt", “polo-shirt", “sweater", “sin- glet",
    “torso-skin", “pants", “shorts/swim-shorts", “skirt", “stock- ings", “socks", “left-boot", “right-boot",
    “left-shoe", “right-shoe", “left- highheel", “right-highheel", “left-sandal", “right-sandal", “left-leg",
    “right-leg", “left-foot", “right-foot", “coat", “dress", “robe", “jumpsuits", “other-full-body-clothes",
    “headwear", “backpack", “ball", “bats", “belt", “bottle", “carrybag", “cases", “sunglasses", “eyewear",
    “gloves", “scarf", “umbrella", “wallet/purse", “watch", “wristband", “tie", “other-accessaries",
    “other-upper-body-clothes", and “other-lower-body-clothes".
    Each instance has a complete set of annotations whenever the corresponding category appears in the current image.

    """
    root = '/data/public/rw/datasets/human/parsing/LV-MHP-v2'
    category = OrderedDict({
        0: 'Background',
        1: 'Cap/hat',
        2: 'Helmet',
        3: 'Face',
        4: 'Hair',
        5: 'Left-arm',
        6: 'Right-arm',
        7: 'Left-hand',
        8: 'Right-hand',
        9: 'Protector',
        10: 'Bikini/bra',
        11: 'Jacket/windbreaker/hoodie',
        12: 'Tee-shirt',
        13: 'Polo-shirt',
        14: 'Sweater',
        15: 'Singlet',
        16: 'Torso-skin',
        17: 'Pants',
        18: 'Shorts/swim-shorts',
        19: 'Skirt',
        20: 'Stockings',
        21: 'Socks',
        22: 'Left-boot',
        23: 'Right-boot',
        24: 'Left-shoe',
        25: 'Right-shoe',
        26: 'Left-highheel',
        27: 'Right-highheel',
        28: 'Left-sandal',
        29: 'Right-sandal',
        30: 'Left-leg',
        31: 'Right-leg',
        32: 'Left-foot',
        33: 'Right-foot',
        34: 'Coat',
        35: 'Dress',
        36: 'Robe',
        37: 'Jumpsuit',
        38: 'Other-full-body-clothes',
        39: 'Headwear',
        40: 'Backpack',
        41: 'Ball',
        42: 'Bats',
        43: 'Belt',
        44: 'Bottle',
        45: 'Carrybag',
        46: 'Cases',
        47: 'Sunglasses',
        48: 'Eyewear',
        49: 'Glove',
        50: 'Scarf',
        51: 'Umbrella',
        52: 'Wallet/purse',
        53: 'Watch',
        54: 'Wristband',
        55: 'Tie',
        56: 'Other-accessary',
        57: 'Other-upper-body-clothes',
        58: 'Other-lower-body-clothes',
    })

    # III. For pose, one image is corresponding to one annotation file with the same prefix.
    # Each annotation file is a "mat" type structure containing labels for all instances for an image,
    # the label for each instance represents:
    keypoints = OrderedDict({
        0: 'Right-ankle',
        1: 'Right-knee',
        2: 'Right-hip',
        3: 'Left-hip',
        4: 'Left-knee',
        5: 'Left-ankle',
        6: 'Pelvis',
        7: 'Thorax',
        8: 'Upper-neck',
        9: 'Head-top',
        10: 'Right-wrist',
        11: 'Right-elbow',
        12: 'Right-shoulder',
        13: 'Left-shoulder',
        14: 'Left-elbow',
        15: 'Left-wrist',
        16: 'Face-bbox-top-left-corner-point',
        17: 'Face-bbox-bottom-right-corner-point',
        18: 'Instance-bbox-top-left-corner-point',
        19: 'Instance-bbox-bottom-right-corner-point',
    })

    def __init__(self, what='train', transforms=None, transform=None, target_transform=None, root=None,
                 which='parsing'):
        root = root or MHPv2.root
        super(MHPv2, self).__init__(root=root, transforms=transforms,
                                    transform=transform, target_transform=target_transform)
        assert what in ('train', 'test', 'val', 'test_top10', 'test_top20')
        assert which in ('parsing', 'pose')

        self.what = what
        self.which = which
        root = Path(root)

        whatinfo = dict(train=('train.txt', 'train'),
                        val=('val.txt', 'val'),
                        test=('test_all.txt', 'test'),
                        test_top10=('test_inter_top10.txt', 'test'),
                        test_top20=('test_inter_top20.txt', 'test'),
                        )
        self.datapath = root / whatinfo[what][1]
        self.imagepath = self.datapath / 'images'
        self.anno_parsing = self.datapath / 'parsing_annos'
        self.anno_pose = self.datapath / 'pose_annos'

        lfile = root / 'list' / whatinfo[what][0]
        with open(lfile, 'r') as f:
            image_ids = [line.strip() for line in f]
        self.image_ids = image_ids
        if what.startswith('test'):
            # annotation 없음
            self.__getitem__ = self._load_image
        if which == 'parsing':
            self._load_anno = self._load_parsing
        elif which == 'pose':
            self._load_anno = self._load_pose

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        i = self.image_ids[index]
        fname = self.imagepath / f'{i}.jpg'
        image = Image.open(fname)
        target = self._load_anno(i)

        return image, target

    def _load_image(self, index):
        i = self.image_ids[index]
        fname = self.imagepath / f'{i}.jpg'
        return Image.open(fname)

    def _load_anno(self, i):
        pass

    def _load_parsing(self, i):
        files = self.anno_parsing.glob(f'{i}_*.png')
        return [Image.open(f).convert('RGB') for f in files]

    def _load_pose(self, i):
        """
        :param i:
        :return:
        """
        f = self.anno_pose / f'{i}.mat'
        mat = scipy.io.loadmat(f)
        keypoints = [v for k, v in mat.items() if k.startswith('person_')]
        # # list of [20 x 3 array] float32,
        # array([[ -1.,  -1.,   2.],
        #        [101., 547.,   0.],
        #        [119., 409.,   0.],
        #        [179., 401.,   0.],
        #        [167., 530.,   0.],
        #        [ -1.,  -1.,   2.],
        #        [157., 408.,   0.],
        #        [161., 238.,   0.],
        #        [155., 126.,   0.],
        #        [152.,  33.,   0.],
        #        [ 87., 407.,   0.],
        #        [ 66., 318.,   0.],
        #        [ 65., 176.,   0.],
        #        [194., 181.,   0.],
        #        [209., 306.,   0.],
        #        [216., 375.,   0.],
        #        [121.,  40.,   0.],
        #        [178., 123.,   0.],
        #        [ 30.,  26.,   0.],
        #        [235., 586.,   0.]], dtype=float32)
        return keypoints

