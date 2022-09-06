## setup.py
from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='gita',
    version='0.1',
    author='Yewon Lim',
    author_email='ga06033@yonsei.ac.kr',
    long_description=long_description,
    include_package_data=False,
    packages=find_packages(exclude=['*dataset*', '*logdir*']),
)
