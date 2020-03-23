
import os
# from distutils.extension import Extension
from setuptools import Extension


ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():
    import numpy

    sources = ["spectrum.pyx"]
    # include_dirs = [numpy.get_include()]
    include_dirs = ['numpy']

    exts = [
        Extension(name='turbustat.simulator.' + os.path.splitext(source)[0],
                  sources=[os.path.join(ROOT, source)],
                  include_dirs=include_dirs)
        for source in sources
    ]

    return exts
