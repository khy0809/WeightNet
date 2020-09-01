import os
from torchvision.datasets import VisionDataset
from PIL import Image
import numpy as np


class LIPpose(VisionDataset):
    """Look Into Person for human parsing
    see : http://sysu-hcp.net/lip/overview.php
    human pose 2d 데이터셋 클래스 (single person)
    - nas 버전
    todo: keypoint 위치가 이미지 사이즈 범위를 벗어나는게 있음. 체크가 필요함

    Label:
        ImageID_PersonId.jpg,
        x1,y1,v1,x2,y2,v2,...x16,y16,v16
        Note: x,y, is the annotation label in (column, row),
              v stands for visuable

        Joint order: (zero indexing)
            0,R_Ankle
            1,R_Knee
            2,R_Hip
            3,L_Hip
            4,L_Knee
            5,L_Ankle
            6,B_Pelvis
            7,B_Spine
            8,B_Neck
            9,B_Head
            10,R_Wrist
            11,R_Elbow
            12,R_Shoulder
            13,L_Shoulder
            14,L_Elbow
            15,L_Wrist
    """
    root = '/data/public/rw/datasets/human/LIP'
    whats = ('train', 'val')
    whatfolders = ('train_images', 'val_images')

    whatfiles = ('lip_train_set.csv', 'lip_val_set.csv')
    labels = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee',
              'L_Ankle', 'B_Pelvis', 'B_Spine', 'B_Neck', 'B_Head',
              'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist')

    def __init__(self, what='train', transforms=None, transform=None, target_transform=None):
        super().__init__(root=LIPpose.root,
                         transforms=transforms, transform=transform, target_transform=target_transform)
        assert what in self.whats, f'{what} is not in {self.whats}'
        self.what = what
        self._whatfolder = os.path.join(self.root, self.whatfolders[self.whats.index(what)])

        self._whatfile = self.whatfiles[self.whats.index(what)]
        self.annotations = self._load_annotation()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        f, k = self.annotations[index]
        f = os.path.join(self._whatfolder, f)
        image = Image.open(f)
        label = k
        return image, label

    def _load_annotation(self):
        file = os.path.join(self.root, 'TrainVal_pose_annotations',
                            self._whatfile)
        with open(file, 'rt') as f:
            return [self._parseline(line) for line in f.readlines()]

    @staticmethod
    def _parseline(line):
        line = line.strip().split(',')
        file = line[0]

        def int_nan(v):
            try:
                return int(v)
            except ValueError:
                return np.nan

        int_nans = tuple(int_nan(v) for v in line[1:])
        # chunk per 3 values
        keypoints = tuple(int_nans[i:i+3] for i in range(0, len(line[1:]), 3))

        return file, keypoints
