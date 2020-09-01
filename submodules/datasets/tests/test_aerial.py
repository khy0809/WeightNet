import datasets
import time


def test_load_RESISC45():
    t0 = time.time()
    dataset = datasets.RESISC45()
    assert len(dataset) == 700 * 45
    print(dataset)
    for i, d in enumerate(dataset):
        print(i, d)
    print(time.time()-t0, ' elapsed')

if __name__ == '__main__':
    test_load_RESISC45()
