#!/usr/bin/env python
# coding=utf-8

# Authors: John Stowers <john@loopbio.com>, Santi Villalba <santi@loopbio.com>, Antonio Ortega <antonio.ortega@kuleuven.be>
# Licence: BSD 3 clause

import os.path as op
from setuptools import setup, find_packages

PACKAGES=find_packages()
print(PACKAGES)
this_directory = op.abspath(op.dirname(__file__))
with open(op.join(this_directory, 'README.md'), 'rb') as f:
   long_description = f.read().decode('UTF-8')

setup(
   name='imgstore-shaliulab',
   license='BSD 3 clause',
   description='IMGStore houses your video frames',
   long_description=long_description,
   long_description_content_type='text/markdown',
   include_package_data=True,
   version='0.4.11',
   url='https://github.com/shaliulab/imgstore',
   author='John Stowers, Santi Villalba, Antonio Ortega',
   author_email='john@loopbio.com, santi@loopbio.com, antonio.ortega@kuleuven.be',
   packages=PACKAGES,
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
       'numpy',
       'pandas',
       'pyyaml',
       'pytz',
       'tzlocal',
       'python-dateutil',
       'confapp-shaliulab==1.1.13',
       'codetiming',
   ],
   tests_require=[
       'pytest',
       'pytest-cov',
       'pytest-pep8'
   ],
   extras_require={
       'bloscpack': ['bloscpack'],
       'cv2cuda': ['cv2cuda>=1.0.4']
   },
   entry_points={
       'console_scripts': [
           'imgstore-view = imgstore.apps:main_viewer',
           'imgstore-save = imgstore.apps:main_saver',
           'imgstore-test = imgstore.apps:main_test',
           'imgstore-codecs = imgstore.apps:list_codecs',
           'imgstore-muxer = imgstore.apps:main_muxer',
           'imgstore2imgstore-muxer = imgstore.apps:imgstore_muxer',
       ]
   },
)
