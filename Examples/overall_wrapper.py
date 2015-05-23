# Licensed under an MIT open source license - see LICENSE


'''

Script for visualizing the output of each statistic in the package.
The two arguments are two folders containing the specific data sets to compare.
'''

import numpy as np
# from turbustat.io import fromfits
import sys
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube, LazyMask
from turbustat.data_reduction import Mask_and_Moments

np.random.seed(248954785)

folder1 = str(sys.argv[1])
folder2 = str(sys.argv[2])

# dataset1 = fromfits(folder1, keywords, verbose=False)
# dataset2 = fromfits(folder2, keywords, verbose=False)

data1, hdr1 = fits.getdata(folder1, header=True)
data2, hdr2 = fits.getdata(folder2, header=True)

# data1 += np.random.normal(0.0, 0.1277369117707014 / 2, data1.shape)
data2 += np.random.normal(0.0, 0.788 / 10, data2.shape)

# cube1 = SpectralCube.read(folder1)
# cube2 = SpectralCube.read(folder2)
cube1 = SpectralCube(data=data1, wcs=WCS(hdr1))
cube1 = cube1.with_mask(LazyMask(np.isfinite, cube1))
cube2 = SpectralCube(data=data2, wcs=WCS(hdr2))
cube2 = cube2.with_mask(LazyMask(np.isfinite, cube2))

set1 = Mask_and_Moments(cube1)
mask = cube1 > 3*set1.scale
set1.make_mask(mask=mask)
set1.make_moments()
set1.make_moment_errors()
dataset1 = set1.to_dict()

set2 = Mask_and_Moments(cube2)
mask = cube2 > 3*set2.scale
set2.make_mask(mask=mask)
set2.make_moments()
set2.make_moment_errors()
dataset2 = set2.to_dict()


# Wavelet Transform

# from turbustat.statistics import Wavelet_Distance

# wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"],
#                                     dataset2["integrated_intensity"]).distance_metric(verbose=True)

# print "Wavelet Distance: %s" % (wavelet_distance.distance)

# # MVC

# from turbustat.statistics import MVC_distance

# mvc_distance = MVC_distance(dataset1, dataset2).distance_metric(verbose=True)

# print "MVC Distance: %s" % (mvc_distance.distance)

# # Spatial Power Spectrum/ Bispectrum

# from turbustat.statistics import PSpec_Distance, BiSpectrum_Distance

# pspec_distance = \
#     PSpec_Distance(dataset1["integrated_intensity"],
#                    dataset2["integrated_intensity"],
#                    weights1=dataset1["integrated_intensity_error"][0]**2.,
#                    weights2=dataset2["integrated_intensity_error"][0]**2.).distance_metric(verbose=True)

# print "Spatial Power Spectrum Distance: %s" % (pspec_distance.distance)

# bispec_distance = BiSpectrum_Distance(dataset1["integrated_intensity"],
#                                       dataset2["integrated_intensity"]).distance_metric(verbose=True)

# print "Bispectrum Distance: %s" % (bispec_distance.distance)

# # Genus

# from turbustat.statistics import GenusDistance

# genus_distance = GenusDistance(dataset1["integrated_intensity"][0],
#                                dataset2["integrated_intensity"][0]).distance_metric(verbose=True)

# print "Genus Distance: %s" % (genus_distance.distance)

# # Delta-Variance

# from turbustat.statistics import DeltaVariance_Distance

# delvar_distance = DeltaVariance_Distance(dataset1["integrated_intensity"],
#                                          dataset2["integrated_intensity"],
#                                          weights1=dataset1[
#                                              "integrated_intensity_error"][0],
#                                          weights2=dataset2["integrated_intensity_error"][0]).distance_metric(verbose=True)

# print "Delta-Variance Distance: %s" % (delvar_distance.distance)

# # VCA/VCS

from turbustat.statistics import VCA_Distance, VCS_Distance

# vcs_distance = VCS_Distance(dataset1["cube"],
#                             dataset2["cube"], breaks=-0.5).distance_metric(verbose=True)

# print "VCS Distance: %s" % (vcs_distance.distance)

vca_distance = VCA_Distance(dataset1["cube"],
                            dataset2["cube"], breaks=1.5).distance_metric(verbose=True)

print "VCA Distance: %s" % (vca_distance.distance)

# Tsallis

from turbustat.statistics import Tsallis_Distance

tsallis_distance = Tsallis_Distance(dataset1["integrated_intensity"][0],
                                    dataset2["integrated_intensity"][0]).distance_metric(verbose=True)

print "Tsallis Distance: %s" % (tsallis_distance.distance)

# High-order stats

from turbustat.statistics import StatMomentsDistance

moment_distance = StatMomentsDistance(dataset1["integrated_intensity"][0],
                                      dataset2["integrated_intensity"][0], 5).distance_metric(verbose=True)

print "Kurtosis Distance: %s" % (moment_distance.kurtosis_distance)

print "Skewness Distance: %s" % (moment_distance.skewness_distance)

# PCA

from turbustat.statistics import PCA_Distance

pca_distance = PCA_Distance(dataset1["cube"][0],
                            dataset2["cube"][0]).distance_metric(verbose=True)

print "PCA Distance: %s" % (pca_distance.distance)

# SCF

from turbustat.statistics import SCF_Distance

scf_distance = SCF_Distance(dataset1["cube"][0],
                            dataset2["cube"][0]).distance_metric(verbose=True)

print "SCF Distance: %s" % (scf_distance.distance)

# Cramer Statistic

from turbustat.statistics import Cramer_Distance

cramer_distance = Cramer_Distance(dataset1["cube"][0],
                                  dataset2["cube"][0]).distance_metric()

print "Cramer Distance: %s" % (cramer_distance.distance)

# Dendrogram Stats

from turbustat.statistics import DendroDistance

dendro_distance = DendroDistance(dataset1["cube"][0],
                                 dataset2["cube"][0]).distance_metric(verbose=True)

print dendro_distance.num_distance
print dendro_distance.histogram_distance

# PDF

from turbustat.statistics import PDF_Distance

pdf_distance = \
    PDF_Distance(dataset1["integrated_intensity"][0],
                 dataset2["integrated_intensity"][0],
                 min_val1=0.05,
                 min_val2=0.05,
                 weights1=dataset1["integrated_intensity_error"][0] ** -2.,
                 weights2=dataset2["integrated_intensity_error"][0] ** -2.)

pdf_distance.distance_metric(verbose=True)

print pdf_distance.distance
