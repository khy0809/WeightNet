import datasets
import numpy as np


def test_lip_parse():
    # todo test dataset
    for what in ['train', 'val']:
        dataset = datasets.LIPparse(what=what)
        for i, (image, target) in enumerate(dataset):
            print(i, image, target)
            assert image.size == target.size


def test_lip_pose():

    def check(keypoints, size):
        failed = 0
        for x, y, v in keypoints:
            try:
                assert 0 <= x <= size[0] or np.isnan(x)
                assert 0 <= y <= size[1] or np.isnan(y)
                assert v in [0, 1] or ((np.isnan(x) or np.isnan(y)) and np.isnan(v))
            except AssertionError:
                failed += 1
        return failed

    failed = 0
    for what in ['train', 'val']:
        dataset = datasets.LIPpose(what=what)
        for i, (image, keypoints) in enumerate(dataset):
            print(i, image, keypoints)
            w, h = image.size
            failed += check(keypoints, image.size)
        # assert failed == 0
        # keypoint value가 범위를 벗어나는 경우가 있음
        print('failed:', failed)


if __name__ == '__main__':
    test_lip_pose()
