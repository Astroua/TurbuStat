
Installing TurbuStat
====================

TurbuStat is currently only available from the `github repo <https://github.com/Astroua/TurbuStat>`_.

TurbuStat requires the follow packages:
 * numpy-1.6
 * matplotlib
 * astropy-1.0
 * scipy-0.12.0
 * sklearn-0.13.0
 * pandas-0.13
 * statsmodels-0.4.0
 * astrodendro-`dev <https://github.com/dendrograms/astrodendro>`_
 * `spectral-cube-v0.2.2 <https://github.com/radio-astro-tools/spectral-cube>`_
 * `signal-id <https://github.com/radio-astro-tools/signal-id>`_

 We recommend installing the `radio_beam <https://github.com/radio-astro-tools/radio_beam>`_ package to aid in the noise estimation in `signal-id <https://github.com/radio-astro-tools/signal-id>`_.

 To install the packages, clone the repository:
 ::
    >>> git clone https://github.com/Astroua/TurbuStat

 Then install the package:
 ::
    >>> python setup.py install

 The script will install numpy and astropy if your python installation does not have them installed. Due to package conflicts, it will **NOT** install the rest of the dependencies! Until this can be fixed, you can check to see if you have all of the dependencies installed by running:
 ::
    >>> python setup.py check_deps

If you find any issues in the installation, please make an `issue on github <https://github.com/Astroua/TurbuStat/issues>`_ or contact the developers at the email on `this page <https://github.com/e-koch>`_. Thank you!
