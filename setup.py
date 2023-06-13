from setuptools import setup

setup(
    name='pycv',
    version='1.2.0',
    packages=['cv3'],
    url='https://github.com/gorodion/pycv',
    license='MIT License',
    author='gorodion',
    author_email='gorodion@bk.ru',
    description='Pythonic cv2',
    install_requires=[
        'numpy>=1.19.5',
        'opencv-python>=4.2.0.34'
    ]
)
