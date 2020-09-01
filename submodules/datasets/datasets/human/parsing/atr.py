from torchvision.datasets import VisionDataset
from datasets.util.tenth import url2Image
import numpy as np


class ATR(VisionDataset):
    """`ATR dataset.
    데이터셋 페이퍼 https://arxiv.org/abs/1503.02391
    @ARTICLE{ATR, author={Xiaodan Liang and Si Liu and Xiaohui Shen and Jianchao Yang and Luoqi Liu and Jian Dong and Liang Lin and Shuicheng Yan}, journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on}, title={Deep Human Parsing with Active Template Regression}, year={2015}, volume={37}, number={12}, pages={2402-2414}, doi={10.1109/TPAMI.2015.2408360}, ISSN={0162-8828}, month={Dec},}

    Args:
        what (string): ('2500', '4565', '997', 'Multi', 'fash', 'dataset10k') 중 혹은 생략시 전체
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    Example:
        .. code:: python
            import datasets as dset
            import torchvision.transforms as transforms
            dataset = dset.ATR()
            print('Number of samples: ', len(dataset))
            img, target = data[3] # load 4th sample
            print("Image Size: ", img.size())
            print("Target Size: ", target.size())
            print(dataset.labels)

    target values (PIL.Image(PNG uint image, NO channel):
        background     0
        hat            1
        hair           2
        sunglass       3
        upper-clothes  4
        skirt          5
        pants          6
        dress          7
        belt           8
        left-shoe      9
        right-shoe     10
        face           11
        left-leg       12
        right-leg      13
        left-arm       14
        right-arm      15
        bag            16
        scarf          17
    """
    _TENTH_ROOT = '/braincloud/datasets/human/ATR/humanparsing'
    _DATA_PATH = f'{_TENTH_ROOT}/JPEGImages'
    _LABEL_PATH = f'{_TENTH_ROOT}/SegmentationClassAug'

    # 데이터 종류별 갯수
    whats = ('2500', '4565', '997', 'Multi', 'fash', 'dataset10k')
    counts = (2476, 3302, 810, 436, 685, 10003)

    # target 어노테이션 레이블 값들 0 ~ 17
    labels = ('background', 'hat', 'hair', 'sunglass', 'upper-clothes', 'skirt',
              'pants', 'dress', 'belt', 'left-shoe', 'right-shoe',
              'face', 'left-leg', 'right-leg', 'left-arm', 'right-arm',
              'bag', 'scarf')

    def __init__(self, what='', transforms=None, transform=None, target_transform=None):
        super().__init__(root=ATR._TENTH_ROOT,
                         transforms=transforms, transform=transform,
                         target_transform=target_transform)
        self.what = what
        if not what:
            self._count = sum(ATR.counts)
            # 전체 데이터셋을 요청할 때 해당 파일 찾기 위한 정보
            self._cumsum = np.cumsum(ATR.counts)
        else:
            what = self._machine_readable(what)
            whats = [w.lower() for w in ATR.whats]
            assert what.lower in whats, f'lower(what) not in {whats}'

            i = whats.index(what.lower())
            self._count = ATR.counts[i]
            self._what = ATR.whats[i]  # 파일명을 생성하기 위힌 값

    def __getitem__(self, index):
        # 파일명이 1로 시작하는 인덱스를 사용함
        if self.what:
            return self._getdata(self.what, index)
        else:
            # whats의 순서대로
            istart = 0
            for i, s in enumerate(self._cumsum):
                if index < s:
                    return self._getdata(ATR.whats[i], index - istart)
                istart = s
        raise IndexError(f'index {index} is out of bound [0, {len(self)})')

    def __len__(self):
        return self._count

    @staticmethod
    def _machine_readable(what):
        if what in ['fashion']:
            return 'fash'
        return what

    def _getdata(self, what, index):
        index += 1
        image = url2Image(f'{ATR._DATA_PATH}/{what}_{index}.jpg')
        target = url2Image(f'{ATR._LABEL_PATH}/{what}_{index}.png')
        if self.transforms:
            return self.transforms(image, target)
        return image, target
