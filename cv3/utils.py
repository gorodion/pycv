def xywh2xyxy(x0, y0, w, h):
    x1 = x0 + w
    y1 = y0 + h
    return x0, y0, x1, y1


def xyxy2xywh(x0, y0, x1, y1):
    w = x1 - x0
    h = y1 - y0
    return x0, y0, w, h


def ccwh2xyxy(xc, yc, w, h):
    x0 = xc - w / 2
    x1 = xc + w / 2
    y0 = yc - h / 2
    y1 = yc + h / 2
    return x0, y0, x1, y1


def xyxy2ccwh(x0, y0, x1, y1):
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    return cx, cy, w, h


def yyxx2xyxy(y0, y1, x0, x1):
    # ¯\_(ツ)_/¯
    return x0, y0, x1, y1

def rel2abs(*coords, width, height):
    '''
    Converts relative coordinates to absolute
    :param coords: iterable (x0, y0, x1, y1, ..., xn, yn)
    :param width:
    :param height:
    :return: iterable
    '''
    assert len(coords) % 2 == 0
    for x, y in zip(*[iter(coords)] * 2):
        yield round(x * width)
        yield round(y * height)


def abs2rel(*coords, width, height):
    '''
    Converts absolute coordinates to relative
    :param coords: iterable (x0, y0, x1, y1, ..., xn, yn)
    :param width:
    :param height:
    :return: iterable
    '''
    assert len(coords) % 2 == 0
    for x, y in zip(*[iter(coords)] * 2):
        yield x / width
        yield y / height
