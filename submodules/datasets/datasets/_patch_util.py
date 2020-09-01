from functools import partial
import inspect
import os


def _has_root_param(f):
    return 'root' in inspect.signature(f).parameters


def _bcloud(packagedict, dataset_module, defaultpath, exceptions=None, root_add_classname=False):
    """
    기존의 dataset 패키지들의 `root` 인자를 브레인클라우드의 특정 위치가 디폴트가 되게 변경해서
    import 시키기 위한 util
    제외할 function들은 inspection으로 혹은 exceptions 목록으로 제외시킴
    """
    exceptions = exceptions or []
    _names = tuple(n for n in dataset_module.__all__ if n not in exceptions)

    _methods = [(n, dataset_module.__dict__[n]) for n in _names
                if _has_root_param(dataset_module.__dict__[n])]

    # _root + dataset클래스명.lower() root가 되게
    if root_add_classname:
        _wraped = {n: partial(f, root=os.path.join(defaultpath, n.lower())) for n, f in _methods}
    else:
        _wraped = {n: partial(f, root=defaultpath) for n, f in _methods}
    _names = list(_wraped.keys())

    packagedict.update(_wraped)

    return _names
