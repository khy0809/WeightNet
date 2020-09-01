import requests
from PIL import Image as _Image
import io


def path2url(path):
    return f'http://gpu-twg.kakaocdn.net{path}'


def url2Image(url):
    """
    tenth url을 받아서 PIL.Imageimage를 돌려준다.
    :param url:
    :return PIL.Image:
    """
    url = path2url(url)
    return _Image.open(requests.get(url, stream=True).raw)


def open(path, mode='r'):
    url = path2url(path)
    r = requests.get(url, stream=True)
    if not r.ok:
        raise ValueError(f'response code: {r.status_code}')
    if mode == 'r':
        return io.StringIO(r.content.decode('utf-8'))
    elif mode == 'rb':
        return io.BytesIO(r.content)
    elif 'w' in mode:
        raise NotImplementedError('really need?')


class Image:

    @classmethod
    def open(cls, path):
        url = path2url(path)
        return _Image.open(requests.get(url, stream=True).raw)
