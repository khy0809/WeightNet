from ._patch_util import _bcloud
try:
    import torchaudio
    _root = '/data/public/rw/datasets/torchaudio'
    __all__ = _bcloud(locals(), torchaudio.datasets, _root)
except ModuleNotFoundError:
    __all__ = []
