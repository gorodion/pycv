def xywh2xyxy(x0, y0, w, h):
    x1 = x0 + w
    y1 = y0 + h
    return x0, y0, x1, y1


def xyxy2xywh(x0, y0, x1, y1):
    assert x1 >= x0 and y1 >= y0
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
    assert x1 >= x0 and y1 >= y0
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    return cx, cy, w, h


def rel2abs(*coords, width, height):
    '''
    :param coords: iterable (x0, y0, x1, y1, ..., xn, yn)
    :param width:
    :param height:
    :return:

    Works only if all the coordinates are relative
    '''
    assert len(coords) % 2 == 0
    if not all(0 <= coord <= 1 for coord in coords): # TODO and float?
        for coord in coords:
            yield coord
        return
    for x, y in zip(*[iter(coords)] * 2):
        yield x * width
        yield y * height