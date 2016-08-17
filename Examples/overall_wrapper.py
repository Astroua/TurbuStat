# Licensed under an MIT open source license - see LICENSE


'''

Script for visualizing the output of each statistic in the package.
The two arguments are two folders containing the specific data sets to compare.
'''

import numpy as np
import sys
import os
from spectral_cube import SpectralCube
from turbustat.data_reduction import Mask_and_Moments

np.random.seed(248954785)

fits1 = str(sys.argv[1])
fits2 = str(sys.argv[2])

# scale = float(sys.argv[3])

cube1 = SpectralCube.read(fits1)
cube2 = SpectralCube.read(fits2)

# Shorten the name for the plots
fits1 = os.path.basename(fits1)
fits2 = os.path.basename(fits2)

# Naive error estimation. Useful for only testing out the methods.
scale1 = cube1.std().value
scale2 = cube2.std().value

set1 = Mask_and_Moments(cube1, scale=scale1)
# mask = cube1 > sigma * set1.scale
# set1.make_mask(mask=mask)
set1.make_moments()
set1.make_moment_errors()
dataset1 = set1.to_dict()

set2 = Mask_and_Moments(cube2, scale=scale2)
# mask = cube2 > sigma * set2.scale
# set2.make_mask(mask=mask)
set2.make_moments()
set2.make_moment_errors()
dataset2 = set2.to_dict()


# Wavelet Transform

from turbustat.statistics import Wavelet_Distance

wavelet_distance = Wavelet_Distance(dataset1["moment0"],
                                    dataset2["moment0"]).distance_metric(verbose=True,
                                                                         label1=fits1,
                                                                         label2=fits2)

print "Wavelet Distance: %s" % (wavelet_distance.distance)

# MVC

from turbustat.statistics import MVC_Distance

mvc_distance = MVC_Distance(dataset1, dataset2).distance_metric(verbose=True,
                                                                label1=fits1,
                                                                label2=fits2)

print "MVC Distance: %s" % (mvc_distance.distance)

# Spatial Power Spectrum/ Bispectrum

from turbustat.statistics import PSpec_Distance, BiSpectrum_Distance

pspec_distance = \
    PSpec_Distance(dataset1["moment0"],
                   dataset2["moment0"],
                   weights1=dataset1["moment0_error"][0]**2.,
                   weights2=dataset2["moment0_error"][0]**2.).distance_metric(verbose=True,
                                                                              label1=fits1,
                                                                              label2=fits2)

print "Spatial Power Spectrum Distance: %s" % (pspec_distance.distance)

bispec_distance = BiSpectrum_Distance(dataset1["moment0"],
                                      dataset2["moment0"]).distance_metric(verbose=True,
                                                                           label1=fits1,
                                                                           label2=fits2)

print "Bispectrum Distance: %s" % (bispec_distance.distance)

# Genus

from turbustat.statistics import GenusDistance

genus_distance = GenusDistance(dataset1["moment0"],
                               dataset2["moment0"]).distance_metric(verbose=True,
                                                                    label1=fits1,
                                                                    label2=fits2)

print "Genus Distance: %s" % (genus_distance.distance)

# Delta-Variance

from turbustat.statistics import DeltaVariance_Distance

delvar_distance = \
    DeltaVariance_Distance(dataset1["moment0"],
                           dataset2["moment0"],
                           weights1=dataset1["moment0_error"][0],
                           weights2=dataset2["moment0_error"][0]).distance_metric(verbose=True,
                                                                                  label1=fits1,
                                                                                  label2=fits2)

print "Delta-Variance Distance: %s" % (delvar_distance.distance)

# VCA/VCS

from turbustat.statistics import VCA_Distance, VCS_Distance

vcs_distance = VCS_Distance(dataset1["cube"],
                            dataset2["cube"], breaks=-0.5).distance_metric(verbose=True,
                                                                           label1=fits1,
                                                                           label2=fits2)

print "VCS Distance: %s" % (vcs_distance.distance)

vca_distance = VCA_Distance(dataset1["cube"],
                            dataset2["cube"], breaks=None).distance_metric(verbose=True,
                                                                           label1=fits1,
                                                                           label2=fits2)

print "VCA Distance: %s" % (vca_distance.distance)

# Tsallis

from turbustat.statistics import Tsallis_Distance

tsallis_distance = Tsallis_Distance(dataset1["moment0"][0],
                                    dataset2["moment0"][0]).distance_metric(verbose=True)

print "Tsallis Distance: %s" % (tsallis_distance.distance)

# High-order stats

from turbustat.statistics import StatMoments_Distance

moment_distance = StatMoments_Distance(dataset1["moment0"],
                                       dataset2["moment0"], 5).distance_metric(verbose=True,
                                                                               label1=fits1,
                                                                               label2=fits2)

print "Kurtosis Distance: %s" % (moment_distance.kurtosis_distance)

print "Skewness Distance: %s" % (moment_distance.skewness_distance)

# PCA

from turbustat.statistics import PCA_Distance

pca_distance = PCA_Distance(dataset1["cube"],
                            dataset2["cube"]).distance_metric(verbose=True,
                                                              label1=fits1,
                                                              label2=fits2)

print "PCA Distance: %s" % (pca_distance.distance)

# SCF

from turbustat.statistics import SCF_Distance

scf_distance = SCF_Distance(dataset1["cube"],
                            dataset2["cube"]).distance_metric(verbose=True,
                                                              label1=fits1,
                                                              label2=fits2)

print "SCF Distance: %s" % (scf_distance.distance)

# Cramer Statistic

from turbustat.statistics import Cramer_Distance

cramer_distance = Cramer_Distance(dataset1["cube"],
                                  dataset2["cube"]).distance_metric()

print "Cramer Distance: %s" % (cramer_distance.distance)

# Dendrogram Stats

from turbustat.statistics import DendroDistance

dendro_distance = DendroDistance(dataset1["cube"],
                                 dataset2["cube"]).distance_metric(verbose=True,
                                                                   label1=fits1,
                                                                   label2=fits2)

print "Dendrogram Number Distance: %s " % (dendro_distance.num_distance)
print "Dendrogram Histogram Distance: %s " % \
    (dendro_distance.histogram_distance)

# PDF

from turbustat.statistics import PDF_Distance

pdf_distance = \
    PDF_Distance(dataset1["moment0"],
                 dataset2["moment0"],
                 min_val1=scale,
                 min_val2=scale,
                 weights1=dataset1["moment0_error"][0] ** -2.,
                 weights2=dataset2["moment0_error"][0] ** -2.)

pdf_distance.distance_metric(verbose=True, label1=fits1, label2=fits2)

print "PDF Hellinger Distance: %s " % (pdf_distance.hellinger_distance)
print "PDF KS-Test Distance: %s " % (pdf_distance.ks_distance)
