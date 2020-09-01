# :cloud: :zap: datasets

- 파이토치 `Dataset` 구현체만 모아놓습니다. 
- public dataset의 `root`의 위치 컨벤션에 따른 디폴트 값 (`/data/public/rw/datasets`)을 적용했습니다.
- 기대효과:
  - nas에 다운로드한 데이터가 중복되지 않을 것이고,
  - 내려받은 데이터 자연스럽게 노티해주게 될 것이고,
  - 누가 새 데이터 올려서 Dataset 클래스 만들어 놓으면 재활용할 수 있을 것이고
  - 이미 다운 받아 쓰고 있는 데이터는 softlink를 잘 활용합시다.

## 설치

- `pip install brain-datasets`

## 사용 예시

```python
import datasets

# https://pytorch.org/docs/stable/torchvision/datasets.html#mnist 의 root 폴더만 nas의 해당 폴더로 변경되어 있습니다.
dataset = datasets.MNIST()
```

## 구현된 `datasets`

- `torchvision.datasets` : https://pytorch.org/docs/stable/torchvision/datasets.html
- `torchaudio.datasets` : https://pytorch.org/audio/datasets.html
- ~~`torchtext.datastes` : https://torchtext.readthedocs.io/en/latest/datasets.html~~

### fashion

- `FashionProductSmall`
- `FashionProduct`

### human parsing

- `ATR`
- `LIPparse`
- `MHPv1`, `MHPv2`

### human pose

- `LIPpose`

### food

- `Food101`

### ImageNet

- `ImageNet` torchvision 보다 로딩 시간 개선

### open images
- `open_images_v5` classification
  - `ImageLevelLabel`
  - `ImageLevelLabelBoxable`

### LVIS dataset

- classification
  - `LvisMultiLabel`

### Places2 dataset

- `Places365`
- `PlacesExtra69`
- `Places434`

### Aerial dataset

- `RESISC45`

### Flower

- `Flowers102`: VGG flower 102 category dataset, consisting of 102 flower categories. Each class consists of between 40 and 258 images.

### Traffic sign

- `GTSRB`: German Traffic Sign Recognition Benchmark

### Medical

- `Aptos2019`: APTOS 2019 Blindness Detection. A large set of retina images taken using fundus photography

### Car

- `CarDataSet` : 2019 3rd ML month with KaKR dataset. 자동차 이미지 데이터셋을 이용한 자동차 차종 분류

### nltk

- [`nltk`](https://www.nltk.org/) 패키지가 설치되어 있다면, `/data/opensets/nltk_data`로 공유된 데이터를 사용합니다.

사용예
```python
In [1]: from datasets.nltk.corpus import wordnet as wn
In [2]: s = wn.synset('wagon.n.01')
In [3]: s
Out[15]: Synset('wagon.n.01')
```
