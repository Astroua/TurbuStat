
'''
Script to run all statistics which use 2D integrated intensity (or
column density, etc...). The terminal arguments are the file names
(FITS files) to compare.
'''

import sys
from astropy.io.fits import getdata

# Format is [0] - data, [1] - header
data1 = getdata(sys.argv[1], header=True)
data2 = getdata(sys.argv[2], header=True)


## Wavelet Transform

from turbustat.statistics import Wavelet_Distance

wavelet_distance = Wavelet_Distance(data1,
                                    data2).distance_metric(verbose=True)

print "Wavelet Distance: %s" % (wavelet_distance.distance)

## Spatial Power Spectrum/ Bispectrum#

from turbustat.statistics import PSpec_Distance, BiSpectrum_Distance

pspec_distance = PSpec_Distance(data1, data2).distance_metric(verbose=True)

print "Spatial Power Spectrum Distance: %s" % (pspec_distance.distance)

bispec_distance = BiSpectrum_Distance(data1,
                                      data2).distance_metric(verbose=True)

print "Bispectrum Distance: %s" % (bispec_distance.distance)

## Genus#

from turbustat.statistics import GenusDistance

genus_distance = GenusDistance(data1[0],
                               data2[0]).distance_metric(verbose=True)

print "Genus Distance: %s" % (genus_distance.distance)

## Delta-Variance

from turbustat.statistics import DeltaVariance_Distance

delvar_distance = DeltaVariance_Distance(data1,
                                         data2).distance_metric(verbose=True)

print "Delta-Variance Distance: %s" % (delvar_distance.distance)

## Tsallis#

from turbustat.statistics import Tsallis_Distance

tsallis_distance= Tsallis_Distance(data1[0],
                                   data2[0]).distance_metric(verbose=True)

print "Tsallis Distance: %s" % (tsallis_distance.distance)

# High-order stats

from turbustat.statistics import StatMomentsDistance

moment_distance = StatMomentsDistance(data1[0],
                                      data2[0]).distance_metric(verbose=True)

print "Kurtosis Distance: %s" % (moment_distance.kurtosis_distance)

print "Skewness Distance: %s" % (moment_distance.skewness_distance)

# ## Dendrogram Stats

from turbustat.statistics import DendroDistance

dendro_distance = DendroDistance(data1[0],
                                 data2[0]).distance_metric(verbose=True)

print dendro_distance.num_distance
print dendro_distance.histogram_distance

# PDF

from turbustat.statistics import PDF_Distance

pdf_distance = \
    PDF_Distance(data1[0],
                 data2[0])

pdf_distance.distance_metric(verbose=False)

print pdf_distance.distance

