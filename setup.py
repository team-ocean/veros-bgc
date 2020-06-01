#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
from codecs import open
import os

import versioneer


CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

INSTALL_REQUIRES = [
    'veros>=0.2.1',
    'graphviz'
]

PACKAGE_DATA = [
    'setup/*/assets.yml', 'setup/*/npzd.yml',
    'setup/*/*.npy', 'setup/*/*.png'
]

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='veros-bgc',
    license='MIT',
    author='Dion Häfner (NBI Copenhagen)',
    author_email='dion.haefner@nbi.ku.dk',
    keywords='oceanography python parallel numpy multi-core '
             'geophysics ocean-model bohrium mpi4py biogeochemistry',
    description='Biogeochemistry plugin for Veros, the versatile ocean simulator.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://veros-bgc.readthedocs.io',
    python_requires='>3.5.2',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    entry_points={
        'veros.setup_dirs': [
            'bgc = veros_bgc.setup'
        ]
    },
    package_data={
        'veros_bgc': PACKAGE_DATA
    },
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    zip_safe=False,
)
