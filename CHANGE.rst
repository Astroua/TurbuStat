
Version 1.0 (unreleased)
------------------------
* #134 - Add rolling back for computing the SCF when shifts are integers (and so don't need the FFT approach)
* #133 - Fix indexing when creating 1D power spectra.
* #131 - Another correction for centroid offset issue. Cleaned moment calcs up a bit.
* #130 - Corrections to moment error calculations.
* #128 - log-log plotting for SCF spectrum. Fitting for SCF spectrum.
* #127 - Standardize all data used for calculating the delta-variance distance. Also allows the `allow_huge` flag to be set in `DeltaVariance` when working on large images (>1 Gb).
* #85 - Added full PCA treatment from the Heyer & Brunt papers. Also implements general handling of physical (given distance) and spectral scale conversions (only working on PCA, needs to be added on to the rest).
* #120 - Minor plotting changes to PDF and bispectrum. Common angular area for Genus normalization. Fix issue with NaNs in the weights for PDFs.
* #118 - Fix SCF surface indexing with non-integer shifts.
* #117 - Use common angular scales for SCF, Moments, Delta-Variance comparison. Dropped using nanmean, nanstd from scipy.stats. Added fourier shifting for SCF.
* #116 - Normalize Genus distance by the image area for comparing unequally-sized images
* #115 - Alter the PCA and Genus metrics to more closely follow the rest of the suite.
* #112 - Stop importing everything at the top level of the module. Also fixes minor input bugs in BiSpectrum_Distance and registers "check_deps" as a command for setup.py.
* #109 - Limit VCA spectrum frequencies below 0.5.
* #106 - Support for multiple data inputs (FITS HDU, SpectralCube, etc). Switched to pytest.
* #105 - Bug in wrapper function. Added test for wrapper.
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