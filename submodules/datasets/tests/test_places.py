import datasets
from PIL import Image


def test_places_standard():
    d = datasets.Places365(split='train', load_pil=False)
    assert len(d) == 1803460
    img, cate = d[0]
    img = Image.open(img)
    assert img.size == (256, 256)

    d = datasets.Places365(split='test', load_pil=False)
    assert len(d) == 328500
    assert d[0][1] == None

    d = datasets.Places365(split='val', load_pil=False)
    assert len(d) == 36500


def test_places_standard_large():
    # The images in the above archives have been resized to have a minimum dimension of 512 while preserving the aspect ratio of the image. 
    d = datasets.Places365(split='train', large=True, load_pil=False)
    assert len(d) == 1803460
    img, cate = d[0]
    img = Image.open(img)
    assert min(img.size) >= 512

    d = datasets.Places365(split='test', large=True, load_pil=False)
    assert len(d) == 328500
    assert d[0][1] == None
    img, cate = d[0]
    img = Image.open(img)
    assert min(img.size) >= 512

    d = datasets.Places365(split='val', large=True, load_pil=False)
    assert len(d) == 36500
    img, cate = d[0]
    img = Image.open(img)
    assert min(img.size) >= 512


def test_places_challenge():
    d = datasets.Places365(split='train', challenge=True, load_pil=False)
    assert len(d) == 8026628
    img, cate = d[0]
    img = Image.open(img)
    assert img.size == (256, 256)
    assert 0 <= cate < 365

    d = datasets.Places365(split='test', challenge=True, load_pil=False)
    assert len(d) == 328500
    assert d[0][1] == None

    d = datasets.Places365(split='val', challenge=True, load_pil=False)
    assert len(d) == 36500


def test_places_challenge_large():
    d = datasets.Places365(split='train', challenge=True, large=True, load_pil=False)
    assert len(d) == 8026628
    img, cate = d[0]
    img = Image.open(img)
    assert 0 <= cate < 365
    assert min(img.size) >= 512

    d = datasets.Places365(split='test', challenge=True, large=True, load_pil=False)
    assert len(d) == 328500
    assert d[0][1] == None
    img, cate = d[0]
    img = Image.open(img)
    assert min(img.size) >= 512

    d = datasets.Places365(split='val', challenge=True, large=True, load_pil=False)
    assert len(d) == 36500
    img, cate = d[0]
    img = Image.open(img)
    assert min(img.size) >= 512


def test_places_extra69():
    larges = [True, False]
    for large in larges:
        d = datasets.PlacesExtra69(split='train', large=large, load_pil=False)
        assert len(d) == 98721
        img, cate = d[0]
        img = Image.open(img)
        if not large:
            assert img.size == (256, 256)
        else:
            assert min(img.size) >= 512
        assert 0 <= cate < 69

        d = datasets.PlacesExtra69(split='test', large=large, load_pil=False)
        assert len(d) == 6600
        img, cate = d[0]
        img = Image.open(img)
        if not large:
            assert img.size == (256, 256)
        else:
            assert min(img.size) >= 512
        # extra69 test셋 target값이 있습니다.
        assert cate is not None


def test_places434():
    larges = [True, False]
    challenges = [True, False]
    
    nstandard = [1803460, 328500, 36500]
    nchallenge = [8026628, 328500, 36500]
    extra = [98721, 6600]

    for large in larges:
        for challenge in challenges:
            d = datasets.Places434(split='train', large=large, challenge=challenge,
                                   load_pil=False)
            if challenge:
                assert len(d) == nchallenge[0] + extra[0]
            else:
                assert len(d) == nstandard[0] + extra[0]
            img, cate = d[0]
            img = Image.open(img)
            if not large:
                assert img.size == (256, 256)
            else:
                assert min(img.size) >= 512
            assert 0 <= cate < 434

            d = datasets.Places434(split='test', large=large, challenge=challenge,
                                   load_pil=False)
            if challenge:
                assert len(d) == nchallenge[1] + extra[1]
            else:
                assert len(d) == nstandard[1] + extra[1]
            img, cate = d[0]
            img = Image.open(img)
            if not large:
                assert img.size == (256, 256)
            else:
                assert min(img.size) >= 512
