
Installing TurbuStat
====================

TurbuStat is currently only available from the [github repo](https://github.com/Astroua/TurbuStat).

TurbuStat requires the follow packages:
 * numpy-1.6
 * matplotlib
 * astropy-1.0
 * scipy-0.12.0
 * sklearn-0.13.0
 * pandas-0.13
 * statsmodels-0.4.0
 * astrodendro-[dev](https://github.com/dendrograms/astrodendro)
 * [spectral-cube-v0.2.2](https://github.com/radio-astro-tools/spectral-cube)
 * [signal-id](https://github.com/radio-astro-tools/signal-id)

 We recommend installing the [radio_beam](https://github.com/radio-astro-tools/radio_beam) package to aid in the noise estimation in [signal-id](https://github.com/radio-astro-tools/signal-id).

 To install the packages, clone the repository:
 >>> git clone https://github.com/Astroua/TurbuStat

 Then install the package:
 >>> python setup.py install

 The script will not install the package if one of the package requirements is not met.
