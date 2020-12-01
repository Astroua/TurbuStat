Version 1.2 (2020-11-30)
------------------------
* #232 - Switch to to github actions
* #228 - Simplify readthedocs builds
* #227 - Fix `binned_statistic` calls for scipy 1.14 (checks for non-finite values).
* #226 - Update package infrastructure following APE 17.

Version 1.1 (2019-11-26)
------------------------
* #217 - Add pip install to docs. Clean-up setup script.

Version 1.0 (2019-04-20)
------------------------
* #216 - Correct the PCA spatial and spectral width definitions. Update unit test data to reflect change.
* #215 - Generalize WCS pixel conversions. Fixed units for the delta-variance break.
* #208 - Fix #207; description of NaN handling for the delta-variance tutorial from @keflavich. **Change to requiring astropy >v2!**
* #171 - Expand PCA tutorial; fix unit conversion error in astropy 3.2dev
* #170 - Add tutorial pages for distance metrics and updated distance metric inputs. **API CHANGES IN DISTANCE METRICS!**
* #192 - Add stddev and weighted fitting for `Wavelet`. Correct bug in fitting weights for `StatisticBase_PSpec2D`.
* #191 - Use robust covariance for linear models. Add residual plots. Add residual bootstrapping for linear models.
* #190 - Add spectral downsampling by averaging for VCA; remove downsampling for VCS.
* #189 - Fixing elliptical p-law parameters for isotropic fields.
* $187 - Correct normalization in 3D power-law fields.
* #186 - Generate mock PPV cubes and 3D power-law fields; added tests for generated power-law in 2D and 3D; renamed `data_reduction` to `moments`; removed masking procedures from `Mask_and_Moments` and renamed to `Moments`
* #185 - Correct SCF weighting for the distance metric. Addresses #184.
* #183 - Add progress bars for long operations. Addresses #180.
* #182 - Separate out plotting functions that were implemented in the `run` functions of the statistics.
* #181 - Add apodizing kernels and beam correction for spatial power-spectrum methods.
* #177 - Add pyfftw as an optional dependency. Allow using pyfftw in all FFT-based statistics for multi-thread support.
* #179 - Remove use of matplotlib._cntr (removed in their version 2.2) for PCA contours. Added scikit-image as a dependency to use their find_contours function.1
* #174 - Set the fraction of valid data points to compute the moments in a region for `StatsMoments`. Updated those test values for the default setting of `min_frac=0.8`.
* #171 - Fix issues with NaN interpolation in Delta-Variance for observational data; added a min. weighting for Delta-Variance; allow segmented fitting for Delta-Variance; added support for weights and WLS in Lm_Seg
* #165 - Added azimuthal constraints when creating 1D power-spectra. Also added radial and azimuthal slicing for the bispectrum. Fixed a critical bug in the bootstrapping for the 2D elliptical power-law model.
* #166 - Removed testing python 2.7 with the dev version of astropy.
* #164 - Add segmented linear fitting for the wavelet transform; updated unit test values to include segmented fitting slopes for wavelets
* #163 - Use Hermitian eigenvalue decomposition for PCA (fixes getting imaginary components); Update the ellipse contour fitting to use the robust fitter from skimage (fixes previous PCA test failures); updated unit test values for the change in PCA
* #161 - python 3 support; use ci-helpers; updated astropy-helpers; consistent convolution results with astropy 2.x; new delta-variance regression test values after convolution updates; DeltaVariance_Distance.curve_distance now restricts to the region between xlow and xhigh
* #159 - Add an elliptical power-law model for fitting 2D power-spectra. Added 2D fitting for PowerSpectrum, MVC, VCA, and SCF, along with tests for each. Unit test values updated to include the 2D slopes. Also updated the tutorials of those 4 methods to demonstrate the new fitting procedure.
* #156 - Complete statistic tutorials; complete angular + physical unit handling; pass break kwargs for all power-spectrum methods; change several APIs to be consistent throughout the package; removed stats_wrapper; added tests to test all cases with a slope and unit conversions; reworked slice thickness choice for VCA and VCS; restrict VCS fitting to real frequencies only; default to periodic boundaries

Version 0.2 (2017-07-03)
-----------
* #147 - Changed Cramer normalization to the spectral norm.
* #146 - Allow weights to be passed in `StatsMoments`.
* #144 - Fix the MVC: subtract mean velocity dispersion, not the array of dispersions.
* #143 - Set periodic boundaries for dendrograms.
* #141 - Added fitting for delta-variance. Added setting of fit limits for delta-variance, wavelets, power spectrum, MVC, and VCA.
* #138 - Add fitting routines and more normalization options for PDFs. Added testing for numpy 1.12 and astropy 1.3.
* #136 - Normalize data to between 0 and 1 for the Cramer statistic.
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