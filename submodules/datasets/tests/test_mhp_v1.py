import datasets
import numpy as np


def test_mhp_v1():
    for what in ['train', 'test']:
        dataset = datasets.MHPv1(what=what)
        max_index = len(dataset.category)
        for i, (image, target) in enumerate(dataset):
            assert len(target) > 0
            print(i, image.size, len(target))
            imsize = image.size

            for t in target:
                assert imsize == t.size
                assert np.asarray(t).max() < max_index
