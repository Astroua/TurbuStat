
'''
Script to run all statistics which use 2D integrated intensity (or
column density, etc...). The terminal arguments are the file names
(FITS files) to compare.
'''

import sys
import os
from astropy.io import fits

fits1 = sys.argv[1]
fits2 = sys.argv[2]

data1 = fits.open(fits1)[0]
data2 = fits.open(fits2)[0]


# Shorten the name for the plots
fits1 = os.path.basename(fits1)
fits2 = os.path.basename(fits2)

# Wavelet Transform

from turbustat.statistics import Wavelet_Distance

wavelet_distance = Wavelet_Distance(data1,
                                    data2).distance_metric(verbose=True,
                                                           label1=fits1,
                                                           label2=fits2)

print "Wavelet Distance: %s" % (wavelet_distance.distance)

# Spatial Power Spectrum/ Bispectrum

from turbustat.statistics import PSpec_Distance, BiSpectrum_Distance

pspec_distance = PSpec_Distance(data1, data2).distance_metric(verbose=True,
                                                              label1=fits1,
                                                              label2=fits2)

print "Spatial Power Spectrum Distance: %s" % (pspec_distance.distance)

bispec_distance = BiSpectrum_Distance(data1,
                                      data2).distance_metric(verbose=True,
                                                             label1=fits1,
                                                             label2=fits2)

print "Bispectrum Distance: %s" % (bispec_distance.distance)

# Genus

from turbustat.statistics import GenusDistance

genus_distance = GenusDistance(data1,
                               data2).distance_metric(verbose=True,
                                                      label1=fits1,
                                                      label2=fits2)

print "Genus Distance: %s" % (genus_distance.distance)

# Delta-Variance

from turbustat.statistics import DeltaVariance_Distance

delvar_distance = DeltaVariance_Distance(data1,
                                         data2).distance_metric(verbose=True,
                                                                label1=fits1,
                                                                label2=fits2)

print "Delta-Variance Distance: %s" % (delvar_distance.distance)

# Tsallis#

from turbustat.statistics import Tsallis_Distance

tsallis_distance= Tsallis_Distance(data1,
                                   data2).distance_metric(verbose=True)

print "Tsallis Distance: %s" % (tsallis_distance.distance)

# High-order stats

from turbustat.statistics import StatMoments_Distance

moment_distance = StatMoments_Distance(data1,
                                       data2).distance_metric(verbose=True,
                                                              label1=fits1,
                                                              label2=fits2)

print "Kurtosis Distance: %s" % (moment_distance.kurtosis_distance)

print "Skewness Distance: %s" % (moment_distance.skewness_distance)

# # Dendrogram Stats

from turbustat.statistics import DendroDistance

dendro_distance = DendroDistance(data1,
                                 data2).distance_metric(verbose=True,
                                                        label1=fits1,
                                                        label2=fits2)

print "Dendrogram Number Distance: %s " % (dendro_distance.num_distance)
print "Dendrogram Histogram Distance: %s " % \
    (dendro_distance.histogram_distance)

# PDF

from turbustat.statistics import PDF_Distance

pdf_distance = \
    PDF_Distance(data1,
                 data2).distance_metric(verbose=True, label1=fits1,
                                        label2=fits2)

print "PDF Hellinger Distance: %s " % (pdf_distance.hellinger_distance)
print "PDF KS-Test Distance: %s " % (pdf_distance.ks_distance)
