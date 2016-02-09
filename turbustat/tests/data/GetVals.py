# Licensed under an MIT open source license - see LICENSE


'''
Save the key results using the testing datasets.
'''

import numpy as np

keywords = ["centroid", "centroid_error", "integrated_intensity",
            "integrated_intensity_error", "linewidth", "linewidth_error",
            "moment0", "moment0_error", "cube"]


from turbustat.tests._testing_data import dataset1, dataset2

## Wavelet Transform

from turbustat.statistics import Wavelet_Distance

wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"],
                                    dataset2["integrated_intensity"]).distance_metric(verbose=False)

wavelet_val = wavelet_distance.wt1.curve

## MVC#

from turbustat.statistics import MVC_distance

mvc_distance = MVC_distance(dataset1, dataset2).distance_metric(verbose=False)

mvc_val = mvc_distance.mvc1.ps1D

## Spatial Power Spectrum/ Bispectrum#

from turbustat.statistics import PSpec_Distance, BiSpectrum_Distance

pspec_distance = PSpec_Distance(dataset1["integrated_intensity"],
                                dataset2["integrated_intensity"],
                                weights1=dataset1["integrated_intensity_error"][0]**2.,
                                weights2=dataset2["integrated_intensity_error"][0]**2.).distance_metric(verbose=False)

pspec_val = pspec_distance.pspec1.ps1D

bispec_distance = BiSpectrum_Distance(dataset1["integrated_intensity"],
                                      dataset2["integrated_intensity"]).distance_metric(verbose=False)

bispec_val = bispec_distance.bispec1.bicoherence

## Genus

from turbustat.statistics import GenusDistance

genus_distance = GenusDistance(dataset1["integrated_intensity"][0],
                               dataset2["integrated_intensity"][0]).distance_metric(verbose=False)

genus_val = genus_distance.genus1.genus_stats

## Delta-Variance

from turbustat.statistics import DeltaVariance_Distance

delvar_distance = \
    DeltaVariance_Distance(dataset1["integrated_intensity"],
                           dataset2["integrated_intensity"],
                           weights1=dataset1["integrated_intensity_error"][0],
                           weights2=dataset2["integrated_intensity_error"][0])

delvar_distance.distance_metric(verbose=False)

delvar_val = delvar_distance.delvar1.delta_var

## VCA/VCS

from turbustat.statistics import VCA_Distance, VCS_Distance

vcs_distance = VCS_Distance(dataset1["cube"],
                            dataset2["cube"]).distance_metric(verbose=False)

vcs_val = vcs_distance.vcs1.ps1D

vca_distance = VCA_Distance(dataset1["cube"],
                            dataset2["cube"]).distance_metric(verbose=False)

vca_val = vca_distance.vca1.ps1D

## Tsallis#

from turbustat.statistics import Tsallis_Distance

tsallis_distance = \
    Tsallis_Distance(dataset1["integrated_intensity"][0],
                     dataset2["integrated_intensity"][0],
                     lags=[1, 2, 4, 8, 16],
                     num_bins=100).distance_metric(verbose=False)

tsallis_val = tsallis_distance.tsallis1.tsallis_fits

# High-order stats

from turbustat.statistics import StatMomentsDistance

moment_distance = \
    StatMomentsDistance(dataset1["integrated_intensity"][0],
                        dataset2["integrated_intensity"][0]).distance_metric(verbose=False)

kurtosis_val = moment_distance.moments1.kurtosis_hist[1]
skewness_val = moment_distance.moments1.skewness_hist[1]

## PCA

from turbustat.statistics import PCA_Distance

pca_distance = PCA_Distance(dataset1["cube"][0],
                            dataset2["cube"][0]).distance_metric(verbose=False)
pca_val = pca_distance.pca1.eigvals

## SCF

from turbustat.statistics import SCF_Distance

scf_distance = SCF_Distance(dataset1["cube"][0],
                            dataset2["cube"][0]).distance_metric(verbose=False)
scf_val = scf_distance.scf1.scf_surface

## Cramer Statistic

from turbustat.statistics import Cramer_Distance

cramer_distance = Cramer_Distance(dataset1["cube"][0],
                                  dataset2["cube"][0]).distance_metric()

cramer_val = cramer_distance.data_matrix1

## Dendrograms

from turbustat.statistics import DendroDistance

min_deltas = np.logspace(-1.5, 0.5, 40)

dendro_distance = DendroDistance(dataset1["cube"][0],
                                 dataset2["cube"][0],
                                 min_deltas=min_deltas).distance_metric(verbose=False)

dendrogram_val = dendro_distance.dendro1.numfeatures

## PDF

from turbustat.statistics import PDF_Distance

pdf_distance = \
    PDF_Distance(dataset1["integrated_intensity"][0],
                 dataset2["integrated_intensity"][0],
                 min_val1=0.05,
                 min_val2=0.05,
                 weights1=dataset1["integrated_intensity_error"][0] ** -2.,
                 weights2=dataset2["integrated_intensity_error"][0] ** -2.)

pdf_distance.distance_metric(verbose=False)

pdf_val = pdf_distance.PDF1.pdf
pdf_ecdf = pdf_distance.PDF1.ecdf
pdf_bins = pdf_distance.bins

np.savez_compressed('checkVals', wavelet_val=wavelet_val,
                    mvc_val=mvc_val,
                    pspec_val=pspec_val,
                    bispec_val=bispec_val,
                    genus_val=genus_val,
                    delvar_val=delvar_val,
                    vcs_val=vcs_val,
                    vca_val=vca_val,
                    tsallis_val=tsallis_val,
                    kurtosis_val=kurtosis_val,
                    skewness_val=skewness_val,
                    pca_val=pca_val,
                    scf_val=scf_val,
                    cramer_val=cramer_val,
                    dendrogram_val=dendrogram_val,
                    pdf_val=pdf_val,
                    pdf_bins=pdf_bins,
                    pdf_ecdf=pdf_ecdf)

np.savez_compressed('computed_distances', mvc_distance=mvc_distance.distance,
                    pca_distance=pca_distance.distance,
                    vca_distance=vca_distance.distance,
                    pspec_distance=pspec_distance.distance,
                    scf_distance=scf_distance.distance,
                    wavelet_distance=wavelet_distance.distance,
                    delvar_distance=delvar_distance.distance,
                    tsallis_distance=tsallis_distance.distance,
                    kurtosis_distance=moment_distance.kurtosis_distance,
                    skewness_distance=moment_distance.skewness_distance,
                    cramer_distance=cramer_distance.distance,
                    genus_distance=genus_distance.distance,
                    vcs_distance=vcs_distance.distance,
                    bispec_distance=bispec_distance.distance,
                    dendrohist_distance=dendro_distance.histogram_distance,
                    dendronum_distance=dendro_distance.num_distance,
                    pdf_hellinger_distance=pdf_distance.hellinger_distance,
                    pdf_ks_distance=pdf_distance.ks_distance)
                    # pdf_ad_distance=pdf_distance.ad_distance)
