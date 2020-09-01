import os
import sys
import time
import pytest_check as check

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import datasets


def test_tencent_ml_classes():
    classes, synsets, hierarchy, subsets = datasets.tencent_ml.TencentML7M.get_classes(lang='ko')
    assert len(classes) == 11166
    assert len(synsets) == 11166
    assert len(hierarchy) == 11166

    classes, synsets, hierarchy, subsets = datasets.tencent_ml.TencentML7M.get_classes()
    assert len(classes) == 11166
    assert len(synsets) == 11166
    assert len(hierarchy) == 11166

    for idx, names in enumerate(classes):
        assert len(names) > 0

    for idx, synset in enumerate(synsets):
        assert len(synset) == len('n00002452')

    for idx, parent_idx in enumerate(hierarchy):
        assert len(classes) > parent_idx


def test_tencent_ml_train():
    t0 = time.time()
    ds = datasets.tencent_ml.TencentML7M('train')
    check.less((time.time() - t0), (60.0 * 5))
    check.equal(len(ds), 916214 + 6607309)

    idx = 0
    image, labels = ds[idx]
    check.equal(image.size, (215, 258))
    check.equal(labels, [2198, 2193, 2188, 2163, 1831, 1054, 1041, 865, 2])

    idx = 458105
    image, labels = ds[idx]
    check.equal(image.size, (500, 375))
    check.equal(labels, [7507, 7460, 7445, 7419, 6526, 6519, 6468, 5174, 5170, 1042, 865, 2])

    idx = 916214 - 1
    image, labels = ds[idx]
    check.equal(image.size, (528, 600))
    check.equal(labels, [7424, 7420, 7418, 6526, 6519, 6468, 5174, 5170, 1042, 865, 2])

    idx = 916214
    image, labels = ds[idx]
    check.equal(image.size, (1024, 768))
    check.equal(labels, [4097, 4089, 4063, 1837, 1054, 1041, 865, 2, 4129, 4132])

    idx = 916214 + 3303654
    image, labels = ds[idx]
    check.equal(image.size, (1024, 681))
    check.equal(labels, [5177, 5170])

    idx = 916214 + 6607309 - 1
    image, labels = ds[idx]
    check.equal(image.size, (1024, 768))
    check.equal(labels, [1193, 1053, 1379])

    # check.equal(len([1 for image, label in ds]), 916214 + 6607309)


def test_tencent_ml_validation():
    t0 = time.time()
    ds = datasets.tencent_ml.TencentML7M('validation')
    check.less((time.time() - t0), 30.0)
    check.equal(len(ds), 4824 + 38739)

    idx = 0
    image, labels = ds[idx]
    check.equal(image.size, (500, 375))
    check.equal(labels, [3371, 2609, 1833, 1054, 1041, 865, 2])

    idx = 4824 // 2 - 1
    image, labels = ds[idx]
    check.equal(image.size, (375, 500))
    check.equal(labels, [2095, 2094, 2092, 2065, 1905, 1829, 1054, 1041, 865, 2])

    idx = 4824 - 1
    image, labels = ds[idx]
    check.equal(image.size, (375, 500))
    check.equal(labels, [4858, 4822, 4781, 4767, 4765, 1067, 1041, 865, 2])

    idx = 4824
    image, labels = ds[idx]
    check.equal(image.size, (1024, 683))
    check.equal(labels, [5173, 5170, 1042, 865, 2, 11026, 892, 890, 884, 870, 859, 5851, 5193, 5181, 9303, 9300, 9289, 1043, 11057, 9305])

    idx = 4824 + 38739 // 2 - 1
    image, labels = ds[idx]
    check.equal(image.size, (1024, 686))
    check.equal(labels, [6866, 6854, 6781, 6767, 6522, 6519, 6468, 5174, 5170, 1042, 865, 2, 6460, 6880, 6878])

    idx = 4824 + 38739 -1
    image, labels = ds[idx]
    check.equal(image.size, (1024, 601))
    check.equal(labels, [5173, 5170, 1042, 865, 2, 5851, 5193, 5181, 1314, 1300, 1192, 1053, 1041])

    # check.equal(len([1 for image, label in ds]), 4824 + 38739)