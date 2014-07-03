TurbuStat
=========

**NOTE - this package is still under development. API may change!**

Statistics of Turbulence
------------------------

This package is aimed at facilitating comparisons between spectral line data
cubes. Included in this package are several techniques described in the literature
which aim to describe some property of a data cube (or its moments/integrated intensity).
We extend these techniques to be used as comparisons.

Distance Metrics
----------------

Ideally, we require a distance metric to satisfy several properties. A full description
is shown in Yeremi et al. (2013).
The key properties are:
*   cubes with similar physics should have a small distance
*   unaffected by coordinate shifts
*   sensitive to differences in physical scale
*   independent of noise levels in the data

Not all of the metrics satisfy the idealized properties. A full description of all
statistics in this package will be shown in Koch et al. (in prep).

Documentation
-------------

*COMING SOON*

Installing
----------

Currently, the only way install TurbuStat is to clone the repository and run
```python
python setup.py install
```
The installation will fail if a dependency is not met. In this case, update or
install the missing package then attempt the install again.


Package Dependencies
--------------------

Requires:

 *   numpy-1.6
 *   matplotlib-1.2
 *   astropy-0.4dev
 *   scipy-0.12
 *   skimage-0.7.1
 *   sklearn-0.13.0
 *   pandas-0.13
 *   statsmodels-0.4.0
 *   astrodendro-dev

Future Dependencies:

 *   [signal-id](https://github.com/radio-astro-tools/signal-id)
 *   [spectral-cube](https://github.com/radio-astro-tools/spectral-cube)

Credits
-------

This package was developed by:

* [Eric Koch](https://github.com/e-koch)
* [Caleb Ward](https://github.com/Astrolebs)
* [Erik Rosolowsky](https://github.com/low-sky)
* Jason Loeppky

Build Status
------------

*COMING SOON*