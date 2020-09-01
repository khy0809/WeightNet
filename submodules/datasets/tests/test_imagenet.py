import os
import sys
import time

import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import datasets


def test_imagenet_train_loading_time():
    t0 = time.time()
    _ = datasets.ImageNet('train')
    assert (time.time() - t0) < 10.0


def test_imagenet_val_loading_time():
    t0 = time.time()
    _ = datasets.ImageNet('val')
    assert (time.time() - t0) < 10.0


def test_imagenet_dataset_count():
    ds = datasets.ImageNet('train')
    assert len(ds) == 1281167

    ds = datasets.ImageNet('val')
    assert len(ds) == 50000
    

def test_imagenet_class_to_idx_train():
    for ds in [datasets.ImageNet('train'), datasets.ImageNet('val')]:
        assert ds.classes[0] == ('tench', 'Tinca tinca')
        assert ds.classes[312] == ('cricket',)
        assert ds.classes[999] == ('toilet tissue', 'toilet paper', 'bathroom tissue')

        assert ds.class_to_idx['tench'] == 0
        assert ds.class_to_idx['Tinca tinca'] == 0

        assert ds.class_to_idx['cricket'] == 312

        assert ds.class_to_idx['toilet tissue'] == 999
        assert ds.class_to_idx['toilet paper'] == 999
        assert ds.class_to_idx['bathroom tissue'] == 999


def test_transforms():
    def transforms(img, label):
        return np.zeros([3, 3, 3]), 1

    ds = datasets.ImageNet('train', transforms=transforms)
    assert (ds[0][0] == np.zeros([3, 3, 3])).all()
    assert ds[0][1] == 1

    ds = datasets.ImageNet('val', transforms=transforms)
    assert (ds[0][0] == np.zeros([3, 3, 3])).all()
    assert ds[0][1] == 1


def test_transform_splits():
    def transform1(img):
        return np.zeros([3, 3, 3])

    def transform2(label):
        return 1

    ds = datasets.ImageNet('train', transform=transform1, target_transform=transform2)
    assert (ds[0][0] == np.zeros([3, 3, 3])).all()
    assert ds[0][1] == 1

    ds = datasets.ImageNet('val', transform=transform1, target_transform=transform2)
    assert (ds[0][0] == np.zeros([3, 3, 3])).all()
    assert ds[0][1] == 1