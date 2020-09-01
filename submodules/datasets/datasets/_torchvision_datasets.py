"""torchvision.datasets 의 `root` 디폴트 위치 채워주기
`root`=`/data/public/rw/datasets/torchvision` 입니다.
- 다른곳에 중복된 데이터가 있다면 `ln -s` 심볼릭 링크를 걸어줍시다.

지원하는 데이터셋은 https://pytorch.org/docs/stable/torchvision/datasets.html

원래 torchvision의 코드를 간단히 사용
```python
# 이런 식의 코드를
import torchvision.datasets as datasets
d = datasets.MNIST(root='./data')

# 이렇게
import datasets
d = datasets.MNIST()

"""
from ._patch_util import _bcloud
try:
    import torchvision
    _root = '/data/public/rw/datasets/torchvision'
    excepts = ['VisionDataset', 'DatasetFolder', 'ImageFolder', 'ImageNet']
    __all__ = _bcloud(locals(), torchvision.datasets, _root, excepts, root_add_classname=False)
except ModuleNotFoundError:
    __all__ = []