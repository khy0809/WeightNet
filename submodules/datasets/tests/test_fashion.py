import datasets
import time


def test_load_fashion_product_small():
    t0 = time.time()
    dataset = datasets.FashionProductSmall()
    print(dataset)
    count = 0
    for i, d in enumerate(dataset):
        print(i, d)
        if d[0].size != (60, 80):
            count += 1
    print(count)
    print(time.time()-t0, ' elapsed')


def test_fashion_product_images():
    dataset = datasets.FashionProduct()
    print(dataset)
    for i, (img, (label, meta)) in enumerate(dataset):
        print(i, img, label, meta.keys())


if __name__ == '__main__':
    test_fashion_product_images()
