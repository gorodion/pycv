import numpy as np

__all__ = [
    'zeros',
    'zeros_like',
    'ones',
    'ones_like',
    'full',
    'full_like',
    'empty',
    'empty_like',
    'white',
    'white_like',
    'black',
    'black_like',
    'random',
    'rand',
    'randn',
    'randint'
]


def zeros(*args):
    return np.zeros(args, np.uint8)


def zeros_like(img):
    return np.zeros_like(img, np.uint8)


def ones(*args):
    return np.ones(args, np.uint8)


def ones_like(img):
    return np.ones_like(img, np.uint8)


def full(*args, value):
    return np.full(args, value, np.uint8)


def full_like(img, value):
    return np.full_like(img, value, np.uint8)


def empty(*args):
    return np.empty(args, np.uint8)


def empty_like(img):
    return np.empty_like(img, np.uint8)


def white(*args):
    return full(*args, value=255)


def white_like(img):
    return full_like(img, value=255)


def random(*args):
    return np.random.randint(0, 256, args, np.uint8)

black = zeros
black_like = zeros_like
rand = randn = randint = random
