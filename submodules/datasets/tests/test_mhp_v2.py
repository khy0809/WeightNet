import datasets
import numpy as np


def test_mhp_v2_parsing():
    whats = ('train', 'test', 'val', 'test_top10', 'test_top20')

    for what in whats:
        dataset = datasets.MHPv2(what=what, which='parsing')
        max_parsing_index = len(dataset.category)
        for i, data in enumerate(dataset):
            if what.startswith('test'):
                assert len(data) == 1
                continue
            assert len(data) == 2
            image, target = data
            assert target
            for t in target:
                a = np.asarray(t)
                # RGB 모드에 R 채널에만 값이 있는 것 같음
                assert a[:, :, 0].max() < max_parsing_index
                assert a[:, :, 1:].max() == 0


def test_mhp_v2_pose():
    whats = ('train', 'test', 'val', 'test_top10', 'test_top20')
    # 이미지 범위를 넘어감 대략 조사
    # SLACK_PIXEL = 30

    for what in whats:
        dataset = datasets.MHPv2(what=what, which='pose')
        shape = (len(dataset.keypoints), 3)

        for i, data in enumerate(dataset):
            if what.startswith('test'):
                assert len(data) == 1
                continue
            assert len(data) == 2
            image, target = data
            # imsz = image.size
            # assert target  # train 1550.mat label 없네
            for t in target:
                assert t.shape == shape
                # print(i, t[:, 0].max(), t[:, 1].max(), imsz)
                # assert t[:, 0].max() < imsz[0] + SLACK_PIXEL
                # assert t[:, 1].max() < imsz[1] + SLACK_PIXEL


if __name__ == '__main__':
    # test_mhp_v2_parsing()
    test_mhp_v2_pose()
