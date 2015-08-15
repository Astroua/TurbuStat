
'''
Compute the dendrograms for the observational cubes, then save the results.
'''

from turbustat.statistics import Dendrogram_Stats
from astropy.io import fits
import numpy as np


# Noise values computed using signal-id (in K)
noise_ngc1333 = 0.128
noise_ophA = 0.252
noise_ic348 = 0.143

min_npix = 100

min_deltas = np.logspace(-0.5, 0.5, 20)

print("Running NGC1333.")

ngc1333_data = fits.getdata("/home/ekoch/sims/complete/ngc1333.13co.fits")
ngc1333 = Dendrogram_Stats(ngc1333_data,
                           min_deltas=min_deltas,
                           dendro_params={"min_npix": 100,
                                          "min_value": noise_ngc1333})

ngc1333.run(save_results=True, output_name="ngc1333.13co_dendrostat.pkl",
            dendro_verbose=True)

print("Running OphA.")

ophA_data = fits.getdata("/home/ekoch/sims/complete/ophA.13co.fits")
ophA = Dendrogram_Stats(ophA_data,
                        min_deltas=min_deltas,
                        dendro_params={"min_npix": 100,
                                       "min_value": noise_ophA})

ophA.run(save_results=True, output_name="ophA.13co_dendrostat.pkl",
         dendro_verbose=True)

print("Running IC 348.")

ic348_data = fits.getdata("/home/ekoch/sims/complete/ic348.13co.fits")
ic348 = Dendrogram_Stats(ic348_data,
                         min_deltas=min_deltas,
                         dendro_params={"min_npix": 100,
                                        "min_value": noise_ic348})

ic348.run(save_results=True, output_name="ic348.13co_dendrostat.pkl",
          dendro_verbose=True)
