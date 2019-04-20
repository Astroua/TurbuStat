#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import sys

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
        from scipy.version import version as sc_version
        if parse_version(sc_version) < parse_version('0.12'):
            print("***Before installing, upgrade scipy to 0.12***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade scipy before installing TurbuStat.")

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
        import astrodendro
    except:
        raise ImportError("Install or upgrade astrodendro to use the"
                          " dendrogram statistics in TurbuStat. "
                          "***NOTE: Need dev version as "
                          "of 17/06/14.***")
    try:
        import spectral_cube
    except ImportError:
        raise ImportError("Install spectral-cube before installing TurbuStat")

    # try:
    #     import signal_id
    # except ImportError:
    #     Warning("signal-id is an optional package for TurbuStat.")


if __name__ == "__main__":

    import ah_bootstrap
    from setuptools import setup

    from setuptools.command.build_ext import build_ext as _build_ext

    class check_deps(_build_ext):
        '''
        Check for the package dependencies.
        '''
        def finalize_options(self):
            _build_ext.finalize_options(self)
            check_dependencies()

    #A dirty hack to get around some early import/configurations ambiguities
    if sys.version_info[0] >= 3:
        import builtins
    else:
        import __builtin__ as builtins
    builtins._ASTROPY_SETUP_ = True

    from astropy_helpers.setup_helpers import (
        register_commands, adjust_compiler, get_debug_option, get_package_info)
    from astropy_helpers.git_helpers import get_git_devstr
    from astropy_helpers.version_helpers import generate_version_py

    # Get some values from the setup.cfg
    try:
        from ConfigParser import ConfigParser
    except ImportError:
        from configparser import ConfigParser

    conf = ConfigParser()
    conf.read(['setup.cfg'])
    metadata = dict(conf.items('metadata'))

    PACKAGENAME = metadata.get('package_name', 'packagename')
    DESCRIPTION = metadata.get('description', 'Astropy affiliated package')
    AUTHOR = metadata.get('author', '')
    AUTHOR_EMAIL = metadata.get('author_email', '')
    LICENSE = metadata.get('license', 'unknown')
    URL = metadata.get('url', 'http://astropy.org')

    # Get the long description from the package's docstring
    __import__(PACKAGENAME)
    package = sys.modules[PACKAGENAME]
    LONG_DESCRIPTION = package.__doc__

    # Store the package name in a built-in variable so it's easy
    # to get from other parts of the setup infrastructure
    builtins._ASTROPY_PACKAGE_NAME_ = PACKAGENAME

    # VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
    VERSION = '1.0.0'

    # Indicates if this version is a release version
    RELEASE = 'dev' not in VERSION

    if not RELEASE:
        VERSION += get_git_devstr(False)

    # Populate the dict of setup command overrides; this should be done before
    # invoking any other functionality from distutils since it can potentially
    # modify distutils' behavior.
    cmdclassd = register_commands(PACKAGENAME, VERSION, RELEASE)
    cmdclassd['check_deps'] = check_deps

    # Adjust the compiler in case the default on this platform is to use a
    # broken one.
    adjust_compiler(PACKAGENAME)

    # Freeze build information in version.py
    generate_version_py(PACKAGENAME, VERSION, RELEASE,
                        get_debug_option(PACKAGENAME))

    # Treat everything in scripts except README.rst as a script to be installed
    scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
               if os.path.basename(fname) != 'README.rst']


    # Get configuration information from all of the various subpackages.
    # See the docstring for setup_helpers.update_package_files for more
    # details.
    package_info = get_package_info()

    # Add the project-global data
    package_info['package_data'].setdefault(PACKAGENAME, [])
    package_info['package_data'][PACKAGENAME].append('data/*')


    # Define entry points for command-line scripts
    entry_points = {'console_scripts': []}

    entry_point_list = conf.items('entry_points')
    for entry_point in entry_point_list:
        entry_points['console_scripts'].append('{0} = {1}'.format(entry_point[0],
                                                                  entry_point[1]))

    # Include all .c files, recursively, including those generated by
    # Cython, since we can not do this in MANIFEST.in with a "dynamic"
    # directory name.
    c_files = []
    for root, dirs, files in os.walk(PACKAGENAME):
        for filename in files:
            if filename.endswith('.c'):
                c_files.append(
                    os.path.join(
                        os.path.relpath(root, PACKAGENAME), filename))
    package_info['package_data'][PACKAGENAME].extend(c_files)

    # Note that requires and provides should not be included in the call to
    # ``setup``, since these are now deprecated. See this link for more details:
    # https://groups.google.com/forum/#!topic/astropy-dev/urYO8ckB2uM

    setup(name=PACKAGENAME,
          version=VERSION,
          description=DESCRIPTION,
          scripts=scripts,
          install_requires=metadata.get('install_requires', 'astropy').strip().split(),
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          cmdclass=cmdclassd,
          zip_safe=False,
          use_2to3=False,
          entry_points=entry_points,
          **package_info
    )
