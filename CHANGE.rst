
Version 1.0 (unreleased)
------------------------
* #86 - Fix ECDFs for standardized data. Added 'find_percentile' and 'find_at_percentile' functions.
* #80 - Disabled Anderson-Darling test for PDFs due to unpredictable errors.
* #79 - Fixed issue in PDF_Distance where all standardized data below 0 was cut-off. Also fixed the plotting in PDF_Distance.
* #78 - Fixed using stddev for the 1D power spectra (MVC, SCF, VCA, PSpec). Pickling for SCF results.