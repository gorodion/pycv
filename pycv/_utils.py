import numpy as np


def _type(img):
    if isinstance(img, list):
        img = np.array(img, 'uint8')
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported type: {type(img)}")
    assert img.ndim == 3 and img.shape[-1] == 3 or img.ndim == 2, f'Incorrect image shape: {img.shape}'  # TODO сравнить условия из scikit-image/matplotlib

    if img.dtype == np.uint8:
        return img
    # if issubclass(img.dtype.type, np.floating):
        # TODO if img.max() <= 1:
        # return img.round().astype('uint8')
    # if issubclass(img.dtype.type, np.integer):
    if issubclass(img.dtype.type, np.number):
        return img.astype('uint8')
    if img.dtype == np.bool:
        return 255 * img.astype('uint8')
    raise TypeError(f"Unsupported dtype: {img.dtype}")

# def _type_decorator
