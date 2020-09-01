import os
from torchvision.datasets import VisionDataset
from PIL import Image


class LvisMultiLabel(VisionDataset):
    """Lvis Dataset - Image-Level Labels

    Args:
        split (str): One of (train, val, test)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    Attributes:
        classes: OrderedDict[index, dict], index to the coco.category information dict (1~1230)
            example) ds[1] = {
                'frequency': 'r', 
                'id': 1, 
                'synset': 'acorn.n.01',
                'image_count': 2,
                'instance_count': 2,
                'synonyms': ['acorn'],
                'def': 'nut from an oak tree',
                'name': 'acorn'}
    """
    root = '/data/opensets/coco'
    splits = ('train', 'val', 'test')
    annoFiles = ('lvis_v0.5_train.json', 'lvis_v0.5_val.json',
                 'lvis_v0.5_image_info_test.json')
    classes = None
    version = '0.5'

    def __init__(self, split='train', transforms=None, transform=None, target_transform=None,
                 root=None):
        from pycocotools.coco import COCO
        root = root or self.root
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        assert split in self.splits, f'{split} is not in {self.splits}'
        self.split = split
        annFile = self.annoFiles[self.splits.index(split)]
        annFile = os.path.join(self.root, 'annotations', annFile)
        self.imgpath = os.path.join(self.root, 'images', f'{split}2017')
        self.imgpath2014 = os.path.join(self.root, 'images', f'{split}2014')
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # the category information
        # Dict[int, object]
        self.classes = self.coco.cats

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]

        fname = img_info['file_name']
        if fname.startswith('COCO_'):
            fname = os.path.join(self.imgpath2014, fname)
        else:
            fname = os.path.join(self.imgpath, fname)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann_data = coco.loadAnns(ann_ids)

        pos_id = list(sorted(set([a['category_id'] for a in ann_data])))
        # testset 은 neg_category_ids가 없음
        neg_id = img_info.get('neg_category_ids', None)

        img = Image.open(fname).convert('RGB')
        target = (pos_id, neg_id) if pos_id or neg_id else None
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @classmethod
    def get_categories(cls):
        """
        get_categories: label strings
        """

        from pycocotools.coco import COCO
        annFile = os.path.join(cls.root, 'annotations',
                               'lvis_v0.5_image_info_test.json')
        coco = COCO(annFile)
        categories = {i: d['name'] for i, d in coco.cats.items()}
        categories[0] = 'not available'
        categories = tuple(categories[i] for i in range(len(categories)))

        return categories
