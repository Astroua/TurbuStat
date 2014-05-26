
'''

Overall TurbuStat package wrapper

Goal is to input two folder names containing data for two runs and compare with all
implemented distance metrics

'''

import numpy as np
from turbustat.io import fromfits
import sys

keywords = {"centroid", "centroid_error", "integrated_intensity", "integrated_intensity_error", "linewidth",\
             "linewidth_error", "moment0", "moment0_error", "cube"}

folder1 = str(sys.argv[1])
folder2 = str(sys.argv[2])

dataset1 = fromfits(folder1, keywords, verbose=False)
dataset2 = fromfits(folder2, keywords, verbose=False)


## Wavelet Transform

from turbustat.statistics import Wavelet_Distance

wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric(verbose=True)

print "Wavelet Distance: %s" % (wavelet_distance.distance)

## MVC#

from turbustat.statistics import MVC_distance

mvc_distance = MVC_distance(dataset1, dataset2).distance_metric(verbose=True)

print "MVC Distance: %s" % (mvc_distance.distance)

## Spatial Power Spectrum/ Bispectrum#

from turbustat.statistics import PSpec_Distance, BiSpectrum_Distance

pspec_distance = PSpec_Distance(dataset1, dataset2).distance_metric(verbose=True)

print "Spatial Power Spectrum Distance: %s" % (pspec_distance.distance)

bispec_distance = BiSpectrum_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric(verbose=True)

print "Bispectrum Distance: %s" % (bispec_distance.distance)

## Genus#

from turbustat.statistics import GenusDistance

genus_distance = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric(verbose=True)

print "Genus Distance: %s" % (genus_distance.distance)

## Delta-Variance

from turbustat.statistics import DeltaVariance_Distance

delvar_distance = DeltaVariance_Distance(dataset1["integrated_intensity"][0],
            dataset1["integrated_intensity_error"][0], dataset2["integrated_intensity"][0],
            dataset2["integrated_intensity_error"][0]).distance_metric(verbose=True)

print "Delta-Variance Distance: %s" % (delvar_distance.distance)

## VCA/VCS#

from turbustat.statistics import VCA_Distance, VCS_Distance

vcs_distance = VCS_Distance(dataset1["cube"],dataset2["cube"]).distance_metric(verbose=True)

print "VCS Distance: %s" % (vcs_distance.distance)

vca_distance = VCA_Distance(dataset1["cube"],dataset2["cube"]).distance_metric(verbose=True)

print "VCA Distance: %s" % (vca_distance.distance)

## Tsallis#

from turbustat.statistics import Tsallis_Distance

tsallis_distance= Tsallis_Distance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric(verbose=True)

print "Tsallis Distance: %s" % (tsallis_distance.distance)

# High-order stats

from turbustat.statistics import StatMomentsDistance

moment_distance = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5).distance_metric(verbose=True)

print "Kurtosis Distance: %s" % (moment_distance.kurtosis_distance)

print "Skewness Distance: %s" % (moment_distance.skewness_distance)

## PCA

from turbustat.statistics import PCA_Distance

pca_distance = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric(verbose=True)

print "PCA Distance: %s" % (pca_distance.distance)

## SCF

from turbustat.statistics import SCF_Distance

scf_distance = SCF_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric(verbose=True)

print "SCF Distance: %s" % (scf_distance.distance)

## Cramer Statistic

from turbustat.statistics import Cramer_Distance

cramer_distance = Cramer_Distance(dataset1["cube"][0], dataset2["cube"][0]).distance_metric()

print "Cramer Distance: %s" % (cramer_distance.distance)

## Dendrogram Stats

from turbustat.statistics import DendroDistance

import os
 ## This one just reads from a saved HDF file, so we need some adjustments first
path1 = "/".join(folder1.split("/")[:-2])
path2 = "/".join(folder2.split("/")[:-2])

file1 = [os.path.join(path1, x) for x in os.listdir(path1) if x[-2:]=="h5"][0]
file2 = [os.path.join(path2,x) for x in os.listdir(path2) if x[-2:]=="h5"][0]
timestep = folder1[-9:-1]

dendro_distance= DendroDistance(file1, file2, timestep).distance_metric(verbose=True)

print dendro_distance.num_distance
print dendro_distance.histogram_distance