"""
discriminative_lexicon_model - Discriminative Lexicon Model in Python
=====================================================================

*discriminative_lexicon_model* provides some matrix-multiplications and accuracy-calculations in the framework of Discriminative Lexicon Model (Baayen et al., 2019).

"""

import os
import sys
import multiprocessing as mp
try:
    from importlib.metadata import requires
except ModuleNotFoundError:
    requires = None
try:
    from packaging.requirements import Requirement
except ModuleNotFoundError:
    Requirement = None

try:
    from . import mapping
    from . import measures
    from . import ldl
except ModuleNotFoundError:
    pass

__author__ = 'Motoki Saito'
__author_email__ = 'motoki.saito@uni-tuebingen.de'
__version__ = '1.4'
__license__ = 'MIT'
__description__ = 'Discriminative Lexicon Model in Python'
__classifiers__ = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering'
]

