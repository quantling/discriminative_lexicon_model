"""
pyldl - Linear Discriminative Learning in Python
=====================================================

*pyldl* provides some matrix-multiplications and accuracy-calculations in the framework of Linear Discriminative Learning (Baayen et al., 2019).

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

__author__ = 'Motoki Saito'
__author_email__ = 'motoki.saito@uni-tuebingen.de'
__version__ = '1.2'
__license__ = 'MIT'
__description__ = 'Linear Discriminative Learning in Python'
__classifiers__ = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering'
]


def sysinfo():
    """
    Prints the system information
    """
    if requires:
        dependencies = [ Requirement(i).name for i in requires('pyndl') if not Requirement(req).marker]

    header = ('Pyldl Information\n'
              '=================\n\n')

    general = ('General Information\n'
               '-------------------\n'
               'Python version: {}\n'
               'Pyldl version: {}\n\n').format(sys.version.split()[0], __version__)

    uname = os.uname()
    osinfo = ('Operating System\n'
              '----------------\n'
              'OS: {s.sysname} {s.machine}\n'
              'Kernel: {s.release}\n'
              'CPU: {cpu_count}\n').format(s=uname, cpu_count=mp.cpu_count())
    osinfo += '\n'

    renames = {'opencv-python':'cv2', 'scikit-learn':'sklearn'}
    excludes = ['python-dateutil', 'threadpoolctl']
    dependencies = [ renames[i] if i in renames.keys() else i for i in dependencies ]
    dependencies = [ i for i in dependencies if not i in excludes ]
    deps = ('Dependencies\n'
            '------------\n')

    if requires:
        deps += "\n".join("{pkg.__name__}: {pkg.__version__}".format(pkg=__import__(dep)) for dep in dependencies)
    else:
        deps = 'You need Python 3.8 or higher to show dependencies.'

    print(header + general + osinfo + deps)
    return None
