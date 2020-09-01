import os
import sys
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import datasets


def check_target_value(ds, dst):
    idx = 0
    image1, (positive_ids, negative_ids) = ds[idx]

    _mid_to_idx = {mid: idx for idx, mid in enumerate(ds.classes)}
    indicies = [_mid_to_idx[mid] for mid in positive_ids if mid in ds.classes]
    image2, labels = dst[idx]
    assert labels.shape == (len(ds.classes), )
    assert (labels[indicies] == 1).all()

    idx = 100
    image1, (positive_ids, negative_ids) = ds[idx]
    indicies = [_mid_to_idx[mid] for mid in positive_ids if mid in ds.classes]
    image2, labels = dst[idx]
    assert labels.shape == (len(ds.classes), )
    assert (labels[indicies] == 1).all()

    idx = -1
    image1, (positive_ids, negative_ids) = ds[idx]
    indicies = [_mid_to_idx[mid] for mid in positive_ids if mid in ds.classes]
    image2, labels = dst[idx]
    assert labels.shape == (len(ds.classes), )
    assert (labels[indicies] == 1).all()


# ImageLevelLabel
def test_open_image_level_label_train():
    t0 = time.time()
    ds = datasets.open_images_v5.ImageLevelLabel('train')
    assert (time.time() - t0) < (60.0 * 2.5)
    assert len(ds) == 5989787

    transforms = datasets.open_images_v5.TargetToVector()
    dst = datasets.open_images_v5.ImageLevelLabel('train', target_transform=transforms)
    check_target_value(ds, dst)


def test_open_image_level_label_val():
    t0 = time.time()
    ds = datasets.open_images_v5.ImageLevelLabel('validation')
    assert (time.time() - t0) < 30.0
    assert len(ds) == 41620

    transforms = datasets.open_images_v5.TargetToVector()
    dst = datasets.open_images_v5.ImageLevelLabel('validation', target_transform=transforms)
    check_target_value(ds, dst)


# ImageLevelLabelBoxable
def test_open_image_level_label_boxable_train():
    t0 = time.time()
    ds = datasets.open_images_v5.ImageLevelLabelBoxable('train')
    assert (time.time() - t0) < (60.0 * 2)
    assert len(ds) == 1743042

    transforms = datasets.open_images_v5.TargetToVector()
    dst = datasets.open_images_v5.ImageLevelLabelBoxable('train', target_transform=transforms)
    check_target_value(ds, dst)


def test_open_image_level_label_boxable_val():
    t0 = time.time()
    ds = datasets.open_images_v5.ImageLevelLabelBoxable('validation')
    assert (time.time() - t0) < 30.0
    assert len(ds) == 37306

    transforms = datasets.open_images_v5.TargetToVector()
    dst = datasets.open_images_v5.ImageLevelLabelBoxable('validation', target_transform=transforms)
    check_target_value(ds, dst)
