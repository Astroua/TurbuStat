
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
save = bool(sys.argv[3])
verbose = bool(sys.argv[4])

dataset1 = fromfits(folder1, keywords, verbose=False)
dataset2 = fromfits(folder2, keywords, verbose=False)


## Wavelet Transform

from turbustat.statistics import Wavelet_Distance

wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric(verbose)

print "Wavelet Distance: %s" % (wavelet_distance.distance)

## MVC#

from turbustat.statistics import MVC_distance

mvc_distance = MVC_distance(dataset1, dataset2).distance_metric(verbose)

print "MVC Distance: %s" % (mvc_distance.distance)

## Spatial Power Spectrum/ Bispectrum#

from turbustat.statistics import PSpec_Distance, BiSpectrum_Distance

pspec_distance = PSpec_Distance(dataset1, dataset2).distance_metric(verbose)

print "Spatial Power Spectrum Distance: %s" % (pspec_distance.distance)

bispec_distance = BiSpectrum_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric(verbose)

print "Bispectrum Distance: %s" % (bispec_distance.distance)

## Genus#

from turbustat.statistics import GenusDistance

genus_distance = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric(verbose)

print "Genus Distance: %s" % (genus_distance.distance)

## Delta-Variance

from turbustat.statistics import DeltaVariance_Distance

delvar_distance = DeltaVariance_Distance(dataset1["integrated_intensity"],
            dataset1["integrated_intensity_error"][0], dataset2["integrated_intensity"],
            dataset2["integrated_intensity_error"][0]).distance_metric(verbose)

print "Delta-Variance Distance: %s" % (delvar_distance.distance)

## VCA/VCS#

from turbustat.statistics import VCA_Distance, VCS_Distance

vcs_distance = VCS_Distance(dataset1["cube"],dataset2["cube"]).distance_metric(verbose)

print "VCS Distance: %s" % (vcs_distance.distance)

vca_distance = VCA_Distance(dataset1["cube"],dataset2["cube"]).distance_metric(verbose)

print "VCA Distance: %s" % (vca_distance.distance)

## Tsallis#

from turbustat.statistics import Tsallis_Distance

tsallis_distance= Tsallis_Distance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric(verbose)

print "Tsallis Distance: %s" % (tsallis_distance.distance)

# High-order stats

from turbustat.statistics import StatMomentsDistance

moment_distance = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5).distance_metric(verbose)

print "Kurtosis Distance: %s" % (moment_distance.kurtosis_distance)

print "Skewness Distance: %s" % (moment_distance.skewness_distance)

## PCA

from turbustat.statistics import PCA_Distance

pca_distance = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric(verbose)

print "PCA Distance: %s" % (pca_distance.distance)

## SCF

from turbustat.statistics import SCF_Distance

scf_distance = SCF_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric(verbose)

print "SCF Distance: %s" % (scf_distance.distance)

## Cramer Statistic

from turbustat.statistics import Cramer_Distance

cramer_distance = Cramer_Distance(dataset1["cube"][0], dataset2["cube"][0]).distance_metric()

print "Cramer Distance: %s" % (cramer_distance.distance)

# ## Dendrogram Stats

from turbustat.statistics import DendroDistance

dendro_distance= DendroDistance(dataset1["cube"][0], dataset2["cube"][0]).distance_metric(verbose)

print dendro_distance.num_distance
print dendro_distance.histogram_distance

if (save):
    np.savez('computed_distances', mvc_distance = mvc_distance.distance, pca_distance = pca_distance.distance, vca_distance = vca_distance.distance, pspec_distance = pspec_distance.distance, scf_distance = scf_distance.distance, wavelet_distance=wavelet_distance.distance, delvar_distance=delvar_distance.distance, tsallis_distance=tsallis_distance.distance, kurtosis_distance=moment_distance.kurtosis_distance, skewness_distance=moment_distance.skewness_distance, cramer_distance=cramer_distance.distance, genus_distance=genus_distance.distance, vcs_distance=vcs_distance.distance, bispec_distance=bispec_distance.distance)
