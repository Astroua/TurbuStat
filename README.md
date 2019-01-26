TurbuStat
=========

[![Build Status](https://travis-ci.org/Astroua/TurbuStat.svg?branch=master)](https://travis-ci.org/Astroua/TurbuStat)

**NOTE - this package is still under development. API may change!**

*See the documentation at [http://turbustat.readthedocs.org/](http://turbustat.readthedocs.org/)*

To be notified of future releases and updates to TurbuStat, please join the mailing list: https://groups.google.com/forum/#!forum/turbustat

Statistics of Turbulence
------------------------

This package is aimed at facilitating comparisons between spectral line data
cubes. Included in this package are several techniques described in the literature
which aim to describe some property of a data cube (or its moments/integrated intensity).
We extend these techniques to be used as comparisons.

Distance Metrics
----------------

Ideally, we require a distance metric to satisfy several properties. A full description
is shown in [Yeremi et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...783...93Y).
The key properties are:
*   cubes with similar physics should have a small distance
*   unaffected by coordinate shifts
*   sensitive to differences in physical scale
*   independent of noise levels in the data

Not all of the metrics satisfy the idealized properties. A full description of all statistics in this package will be shown in Koch et al. (submitted). The paper results can be reproduced using the scripts in [AstroStat_Results](https://github.com/Astroua/AstroStat_Results).

Installing
----------

Currently, the only way install TurbuStat is to clone the repository and run
```python
python setup.py install
```

Package Dependencies
--------------------

Requires:

 *   astropy>=2.0
 *   numpy>=1.7
 *   matplotlib>=1.2
 *   scipy>=0.12
 *   sklearn>=0.13.0
 *   statsmodels>=0.4.0
 *   scikit-image>=0.12

Recommended:

 *   [spectral-cube](https://github.com/radio-astro-tools/spectral-cube) - Efficient handling of PPV cubes. Required for calculating moment arrays in `turbustat.data_reduction.Mask_and_Moments`
 *   [astrodendro-development](https://github.com/dendrograms/astrodendro) - Required for calculating dendrograms in `turbustat.statistics.dendrograms`
 *   [radio_beam](https://github.com/radio-astro-tools/radio_beam) - A class for handling radio beams and useful utilities. Required for correcting for the beam shape in spatial power spectra. Automatically installed with spectral-cube.

Optional:
 *   [emcee](http://dan.iel.fm/emcee/current/) - Affine Invariant MCMC. Used for fitting the size-line width relation in PCA and fitting PDFs.

Credits
-------

If you make use of this package in a publication, please cite our accompanying paper:

```
UPCOMING
```

A description of the distance metrics is provided in this paper:
```
@ARTICLE{Koch2017,
   author = {{Koch}, E.~W. and {Ward}, C.~G. and {Offner}, S. and {Loeppky}, J.~L. and 
	{Rosolowsky}, E.~W.},
    title = "{Identifying tools for comparing simulations and observations of spectral-line data cubes}",
  journal = {\mnras},
archivePrefix = "arXiv",
   eprint = {1707.05415},
 keywords = {methods: statistical, ISM: clouds, radio lines: ISM},
     year = 2017,
    month = oct,
   volume = 471,
    pages = {1506-1530},
      doi = {10.1093/mnras/stx1671},
   adsurl = {http://adsabs.harvard.edu/abs/2017MNRAS.471.1506K},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
Citations courtesy of [ADS](https://ui.adsabs.harvard.edu/#).
