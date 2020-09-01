# Tencent ML-Images


- https://arxiv.org/abs/1901.01703

## 이 데이터 정리과정
- `ImageNet-11k` 
  - 11,797,630 training images, 11,221 category에서
  - 11,221에서 1,989 분류 제외 _event_, _summer_ 등 매우 추상적인 것 제외
  - 10,322,935 images of 9,232 categories left
  - _dog_ 가 있다면, _Husky_ 등 800 finer-grained 분류와 해당 이미지 추가 
    - from the whole vocabulary of ImageNet
  - 여기까지 해서 10,756,941 images, covering `10,032` categories from ImageNet
- `Open-Images-v1`
  -  9M images, 6K categories 에서
  - 650 장 이미지 이하의 카테고리 및 이미지 제외
  - 이미지넷에서 처럼 visual domain 관점에서 추상적인 분류 삭제
  - 위의 10,032 categories와 중복이거나, 유의어 해당하는 것들은 위의 Imagenet-11k 분류로 merge
  - 만약, 제거했더니 tag가 없는 이미지는 제거
  - 결국 6,902,811 training images and 38,739 validation images are remained, 
    - covering `1,134` unique categories
- `Tencent MLImages`
  - 위의 두 작업을 merge해서 
  - `17,609,752` training and `88,739` validation images, 
    - covering `11,166` categories. (10,032 + 1,134)
- Tag augmentation
  - ImageNet-11k 가 single 레이블인 것을 늘리는 작업
  - 수작업으로 하는 것은 어렵다고 판단
  - WordNet의 wordid로 `semantic hierarchy among these 11,166`를 만들고
    - root는 4개의 tree가됨 _thing_, _matter_, _physical object_ and _atmospheric phenomenon_
    - 제일 깊은 depth는 16, 평균 7.47 depth
  - 만든 카테고리 tree구조에서 모든 ancestor들에 positive tag들을 추가 (_dog_ 있으면 _animal_도 마킹)
- Imagenet-11k와 open-images 분류간 co-occurrence 이용
  - open images 데이터로 resnet-101 학습 `1,134` 아웃풋을 내는 모델
  - ImageNet11k의 이미지에 위 모델을 적용
  - posterior p > 0.95 이상의 category는 positive 태깅 (즉, ImageNet-11K에 open-images만의 태그가 추가됨)
  - ImageNet-11k의 _i_카테고리와 open-images _j_ 카테고리간의 co-occurrence 비율 계산 
  - semantic hierarchy의 path가 없는데도, _CO(i, j)_ > 0.5 인 것들의 처리
    - 강한 연관이 있다고 보고, _i_는 모두 _j_ tag도 추가
    - 예를 들면 _sea snake_ 원래 annotation 되어 있는 이미지에 _sea_ tag도 추가
