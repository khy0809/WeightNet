import os
import sys
import time

import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import datasets


def test_imagenet_bbox():
    ds = datasets.ImageNetLoc('train')
    image, bbox, label = ds[0]
    
    assert image.width == 250
    assert image.height == 250
    assert bbox == [[0, 0, 250, 250]]
    assert label == 0

    image, bbox, label = ds[7000]

    assert image.width == 358
    assert image.height == 500
    assert bbox == [[66, 165, 239, 155]]
    assert label == 5


def test_imagenet_loc_covered():
    ds = datasets.ImageNetLocCovered('train')
    image, label = ds[0]

    assert image.width < 250
    assert image.height < 250
    assert label == 0

    image, label = ds[7000]

    assert image.width < 500
    assert image.height < 500
    assert label == 5


def test_imagenet_covered():
    ds = datasets.ImageNetLocCovered('train')
    image, label = ds[0]

    assert image.width < 250
    assert image.height < 250
    assert label == 0

    image, label = ds[7000]

    assert image.width < 500
    assert image.height < 500
    assert label == 5
