#!/usr/bin/env python

from ez_setup import use_setuptools
use_setuptools()


from setuptools import setup, find_packages
from pkg_resources import parse_version


def check_dependencies():
    '''
    setuptools causes problems for installing packages (especially
    statsmodels). Use this function to abort installation instead.
    '''

    try:
        import cython
    except ImportError:
        raise ImportError("Install cython before installing TurbuStat.")

    try:
        import matplotlib
        mpl_version = matplotlib.__version__
        if parse_version(mpl_version) < parse_version('1.2'):
            print("***Before installing, upgrade matplotlib to 1.2***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade matplotlib before installing TurbuStat.")

    try:
        from numpy.version import version as np_version
        if parse_version(np_version) < parse_version('1.6'):
            print("***Before installing, upgrade numpy to 1.6***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade numpy before installing TurbuStat.")

    try:
        from scipy.version import version as sc_version
        if parse_version(sc_version) < parse_version('0.12'):
            print("***Before installing, upgrade scipy to 0.12***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade scipy before installing TurbuStat.")

    try:
        from pandas.version import version as pa_version
        if parse_version(pa_version) < parse_version('0.13'):
            print("***Before installing, upgrade pandas to 0.13***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade pandas before installing TurbuStat.")

    try:
        from statsmodels.version import version as sm_version
        if parse_version(sm_version) < parse_version('0.4.0'):
            print("***Before installing, upgrade statsmodels to 0.4.0***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade statsmodels before installing TurbuStat.")

    try:
        import sklearn
        skl_version = sklearn.__version__
        if parse_version(skl_version) < parse_version('0.13.0'):
            print("***Before installing, upgrade sklearn to 0.13.0***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade sklearn before installing TurbuStat.")

    try:
        from astropy.version import version as ast_version
        if parse_version(ast_version[:3]) < parse_version('0.4'):
            print(("""***Before installing, upgrade astropy to 0.4.
                    NOTE: This is the dev version as of 17/06/14.***"""))
            raise ImportError("")
    except:
        raise ImportError(
            "Install or upgrade astropy before installing TurbuStat.")

    try:
        import astrodendro
    except:
        raise ImportError(("""Install or upgrade astrodendro before installing
                            TurbuStat. ***NOTE: Need dev version as
                            of 17/06/14.***"""))

if __name__ == "__main__":

    check_dependencies()

    setup(name='turbustat',
          version='0.0',
          description='Distance metrics for comparing spectral line data cubes.',
          author='Eric Koch, Caleb Ward, Jason Loeppky and Erik Rosolowsky',
          author_email='koch.eric.w@gmail.com',
          url='http://github.com/Astroua/TurbuStat',
          scripts=[],
          packages=find_packages(
              exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
          )
