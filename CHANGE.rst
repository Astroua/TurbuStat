
Version 1.0 (unreleased)
------------------------
* #102 - Homogenized the structure of the statistics and distance metrics. Several API changes and two metrics had their names changed: StatMomentDistance -> StatMoment_Distance and MVC_distance -> MVC_Distance.
* #104 - Common base class for 2D power spectra
* #103 - implement a corrected version of the wavelet transform
* #101 - Fixed some bugs brought in with #92 for segmented fitting. Using OLS instead of WLS.
* #100 - Fix plotting in PDF_Distance
* #99 - Fixed moment array loading when a full path to the cube is given.
* #98 - Removed scripts for the paper results from TurbuStat. Now available in a separate repo.
* #96 - Fix error calculation in Delta Variance
* #95 - remove double rounding and clean-up unused code in Bispectrum
* #93 - Updated this file :).
* #92 - Re-wrote pspec to use binned_statistic; now returns the proper spatial frequencies
* #89 - Make signal_id and astrodendro optional packages
* #86 - Fix ECDFs for standardized data. Added 'find_percentile' and 'find_at_percentile' functions.
* #80 - Disabled Anderson-Darling test for PDFs due to unpredictable errors.
* #79 - Fixed issue in PDF_Distance where all standardized data below 0 was cut-off. Also fixed the plotting in PDF_Distance.
* #78 - Fixed using stddev for the 1D power spectra (MVC, SCF, VCA, PSpec). Pickling for SCF results.