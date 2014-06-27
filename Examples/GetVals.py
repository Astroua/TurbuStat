
'''

Overall TurbuStat package wrapper

Goal is to input two folder names containing data for two runs and compare with all
implemented distance metrics

'''

import numpy as np
from turbustat.io import fromfits
import sys

keywords = {"centroid", "centroid_error", "integrated_intensity", "integrated_intensity_error", "linewidth", "linewidth_error", "moment0", "moment0_error", "cube"}

folder1 = "/srv/astro/erickoch/enzo_sims/frac_factorial_set/Fiducial128_1.0.0/Fiducial128_1_21_0_0_flatrho_0021_13co/"

dataset1 = fromfits(folder1, keywords)

folder2 = "/srv/astro/erickoch/enzo_sims/frac_factorial_set/Design4.0.0/Design4_21_0_0_flatrho_0021_13co/"

dataset2 = fromfits(folder2, keywords)


## Wavelet Transform

from turbustat.statistics import wt2D

wavelet_val =  wt2D(dataset1["integrated_intensity"][0], np.logspace(np.log10(round((5. / 3.), 3)), np.log10(min(dataset1["integrated_intensity"][0].shape)), 50))
wavelet_val.run()
wavelet_val = wavelet_val.Wf
        
## MVC#

from turbustat.statistics import MVC

mvc_val = MVC(dataset1["centroid"][0], dataset1["moment0"][0], dataset1["linewidth"][0], dataset1["centroid"][1])
mvc_val = mvc_val.run()
mvc_val = mvc_val.ps1D

## Spatial Power Spectrum/ Bispectrum#

from turbustat.statistics import PSpec_Distance, BiSpectrum_Distance

pspec_val = PSpec_Distance(dataset1, dataset2).pspec1.ps1D

bispec_val = BiSpectrum_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).bispec1.bicoherence

## Genus#

from turbustat.statistics import Genus

genus_val = Genus(dataset1["integrated_intensity"][0])
genus_val = genus_val.run()
genus_val = genus_val.thresholds

## Delta-Variance

from turbustat.statistics import DeltaVariance_Distance

delvar_val = DeltaVariance_Distance(dataset1["integrated_intensity"],
            dataset1["integrated_intensity_error"][0], dataset2["integrated_intensity"],
            dataset2["integrated_intensity_error"][0]).delvar1.delta_var
## VCA/VCS#

from turbustat.statistics import VCA_Distance, VCS_Distance, VCS

vcs_val = VCS(dataset1["cube"][0],dataset1["cube"][1])
vcs_val = vcs_val.run()
vcs_val = vcs_val.vel_freqs

vca_val = VCA_Distance(dataset1["cube"],dataset2["cube"]).vca1.ps1D

## Tsallis#

from turbustat.statistics import Tsallis_Distance

tsallis_val = Tsallis_Distance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).tsallis1.tsallis_fits

# High-order stats

from turbustat.statistics import StatMomentsDistance

kurtosis_val = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5).moments1.kurtosis_array 
skewness_val = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5).moments1.skewness_array 

## PCA

from turbustat.statistics import PCA_Distance

pca_val = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0]).pca1.eigvals

## SCF

from turbustat.statistics import SCF_Distance

scf_val = SCF_Distance(dataset1["cube"][0],dataset2["cube"][0]).scf1.scf_surface

## Cramer Statistic

from turbustat.statistics import Cramer_Distance

cramer_val = Cramer_Distance(dataset1["cube"][0], dataset2["cube"][0]).data_matrix1

np.savez('checkVals', wavelet_val=wavelet_val, mvc_val=mvc_val, pspec_val=pspec_val, bispec_val = bispec_val, genus_val=genus_val, delvar_val=delvar_val, vcs_val=vcs_val, vca_val=vca_val, tsallis_val=tsallis_val, kurtosis_val=kurtosis_val, skewness_val=skewness_val, pca_val=pca_val, scf_val=scf_val, cramer_val=cramer_val)