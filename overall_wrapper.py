
'''

Overall TurbuStat package wrapper

Goal is to input two folder names containing data for two runs and compare with all
implemented distance metrics

'''

import numpy as np
from utilities import fromfits
import sys

keywords = {"centroid", "centroid_error", "integrated_intensity", "integrated_intensity_error", "linewidth",\
             "linewidth_error", "moment0", "moment0_error", "cube"}

folder1 = str(sys.argv[1])
folder2 = str(sys.argv[2])

dataset1 = fromfits(folder1, keywords, verbose=False)
dataset2 = fromfits(folder2, keywords, verbose=False)


## Wavelet Transform

from wavelets import Wavelet_Distance

wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric(verbose=True)

print "Wavelet Distance: %s" % (wavelet_distance.distance)

## MVC

from mvc import MVC_distance

mvc_distance = MVC_distance(dataset1, dataset2).distance_metric(verbose=True)

print "MVC Distance: %s" % (mvc_distance.distance)

## Spatial Power Spectrum/ Bispectrum

from pspec_bispec import PSpec_Distance, BiSpec_Distance

pspec_distance = PSpec_Distance(dataset1, dataset2).distance_metric(verbose=True)

print "Spatial Power Spectrum Distance: %s" % (pspec_distance.distance)

# bispec_distance = BiSpec_Distance()

# print " Bispectrum Distance: %s" % (bispec_distance.distance)

## Genus

from genus import GenusDistance

genus_distance = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric(verbose=True)

print "Genus Distance: %s" % (genus_distance.distance)

## Delta-Variance

from delta_variance import DeltaVariance_Distance

delvar_distance = DeltaVariance_Distance(dataset1["integrated_intensity"][0],
            dataset1["integrated_intensity_error"][0], dataset2["integrated_intensity"][0],
            dataset2["integrated_intensity_error"][0]).distance_metric(verbose=True)

print "Delta-Variance Distance: %s" % (delvar_distance.distance)

## VCA/VCS

from vca_vcs import VCA_Distance, VCS_Distance

vcs_distance = VCS_Distance(dataset1["cube"],dataset2["cube"]).distance_metric(verbose=True)

print "VCS Distance: %s" % (vcs_distance.distance)

vca_distance = VCA_Distance(dataset1["cube"],dataset2["cube"]).distance_metric(verbose=True)

print "VCA Distance: %s" % (vca_distance.distance)

## Tsallis

from tsallis import Tsallis_Distance

tsallis_distance= Tsallis_Distance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric(verbose=True)

print "Tsallis Distance: %s" % (tsallis_distance.distance)

## High-order stats

from stat_moments import StatMomentsDistance

moment_distance = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5).distance_metric(verbose=True)

print "Kurtosis Distance: %s" % (moment_distance.kurtosis_distance)

print "Skewness Distance: %s" % (moment_distance.skewness_distance)

## PCA

from pca import PCA_Distance

pca_distance = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric(verbose=True)

print "PCA Distance: %s" % (pca_distance.distance)

## SCF

from scf import SCF_Distance

scf_distance = SCF_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric(verbose=True)

print "SCF Distance: %s" % (scf_distance.distance)
