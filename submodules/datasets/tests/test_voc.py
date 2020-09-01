import datasets
import numpy as np
from torch.utils.data.dataloader import DataLoader


def test_voc():
    image_sets = ['train', 'trainval', 'val']
    for image_set in image_sets:
        dataset = datasets.VOCSegmentation(download=True, image_set=image_set)
        # loader = DataLoader(datasets)
        # for data in loader:
        #     pass
        # DataLoader
        dataset = datasets.VOCDetection(download=True, image_set=image_set)
        loader = DataLoader(datasets)
        for data in loader:
            pass
        # datasets = datasets.Cityscapes(download=True, image_set=image_set)


if __name__ == '__main__':
    test_voc()
