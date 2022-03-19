import warnings
import numpy as np

warnings.simplefilter('always', UserWarning)


def typeit(img):
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img
    warnings.warn('The image was copied because it needs to be cast to the correct type. To avoid copying, please cast the image to np.ndarray with np.uint8 dtype')
    return np.array(img).astype('uint8')
    # if isinstance(img, list):
    #     img = np.array(img, 'uint8')
    # if not isinstance(img, np.ndarray):
    #     raise TypeError(f"Unsupported type: {type(img)}")
    # assert img.ndim == 3 and img.shape[-1] == 3 or img.ndim == 2, f'Incorrect image shape: {img.shape}'
    # if img.dtype == np.uint8:
    #     return img
    # if issubclass(img.dtype.type, np.floating):
        # TODO if img.max() <= 1:
        # return img.round().astype('uint8')
    # if issubclass(img.dtype.type, np.integer):
    # if issubclass(img.dtype.type, np.number):
    #     return img.astype('uint8')
    # if img.dtype == np.bool:
    #     return 255 * img.astype('uint8')
    # raise TypeError(f"Unsupported dtype: {img.dtype}")


def type_decorator(func):
    def wrapper(img, *args, **kwargs):
        img = typeit(img)
        return func(img, *args, **kwargs)
    return wrapper