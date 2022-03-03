from setuptools import setup

setup(
    name='cv3',
    version='1.0',
    packages=['pycv'],
    url='https://github.com/gorodion/cv3',
    license='GNU Lesser General Public License v2.1',
    author='gorodion',
    author_email='gorodion@bk.ru',
    description='Pythonic cv2',
    install_requires=[
        'numpy>=1.19.5',
        'opencv-python>=4.2.0.34'
    ]
)
