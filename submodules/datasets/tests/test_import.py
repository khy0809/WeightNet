
def test_module_import():
    import datasets

    m = datasets.MNIST()
    # m2 = torchvision.datasets.MNIST()
    # assert len(m) == len(m2)
    assert len(m) == 60000
import torch
torch.save