#!/usr/bin/env python
# coding=utf-8

# Authors: John Stowers <john@loopbio.com>, Santi Villalba <santi@loopbio.com>
# Licence: BSD 3 clause

import os.path as op
from setuptools import setup, find_packages

this_directory = op.abspath(op.dirname(__file__))
with open(op.join(this_directory, 'README.md'), 'rb') as f:
    long_description = f.read().decode('UTF-8')



# attention. you need to update the numbers ALSO in the imgstore/__init__.py file
version = "0.3.1" 
with open("imgstore/_version.py", "w") as fh:
    fh.write(f"__version__ = '{version}'\n")

packages = find_packages()
print(packages)

setup(
    name='imgstore',
    license='BSD 3 clause',
    description='IMGStore houses your video frames',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    version=version,
    url='https://github.com/loopbio/imgstore',
    author='John Stowers, Santi Villalba',
    author_email='john@loopbio.com, santi@loopbio.com',
    packages=packages,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'opencv-python',
        'numpy',
        'pandas<=1.2.4',
        'pyyaml',
        'pytz',
        'tzlocal',
        'python-dateutil',
        'gitpython',
        'shapely>=1.7.1',
        'descartes',
        'tqdm',
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-pep8'
    ],
    extras_require={
        'bloscpack': ['bloscpack']
    },
    entry_points={
        'console_scripts': [
            'imgstore-sensor = imgstore.apps:main_sensor',
            'imgstore-view = imgstore.apps:main_viewer',
            'imgstore-save = imgstore.apps:main_saver',
            'imgstore-test = imgstore.apps:main_test',
            'multistore-index = imgstore.apps:main_multistore_index',
            'clip-chunk = imgstore.apps:clip_chunk',
        ]
    },
)
