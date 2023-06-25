from setuptools import setup

setup(
    name='pycv',
    version='1.2.1',
    packages=['cv3'],
    url='https://github.com/gorodion/pycv',
    license='Apache 2.0',
    author='gorodion',
    author_email='domonion@list.ru',
    description='Pythonic cv2',
    install_requires=[
        'numpy>=1.19.5',
        'opencv-python>=4.2.0.34'
    ]
)
