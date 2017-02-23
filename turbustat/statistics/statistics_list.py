# Licensed under an MIT open source license - see LICENSE

'''
Returns a list of all available distance metrics
'''

statistics_list = ["Wavelet", "MVC", "PSpec", "Bispectrum",
                   "DeltaVariance_Curve", "DeltaVariance_Slope",
                   "Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF", "Cramer",
                   "Skewness", "Kurtosis", "VCS_Small_Scale", "VCS_Break",
                   "VCS_Large_Scale", "PDF_Hellinger", "PDF_KS",
                   "PDF_Lognormal",  # "PDF_AD",
                   "Dendrogram_Hist", "Dendrogram_Num"]

twoD_statistics_list = \
    ["Wavelet", "MVC", "PSpec", "Bispectrum", "DeltaVariance",
     "Genus", "Tsallis", "Skewness", "Kurtosis",
     "PDF_Hellinger", "PDF_KS",
     "PDF_Lognormal", # "PDF_AD",
     "Dendrogram_Hist", "Dendrogram_Num"]

threeD_statistics_list = \
    ["VCS", "VCA", "PCA", "SCF", "Cramer", "VCS_Small_Scale",
     "VCS_Large_Scale", "VCS_Break", "PDF_Hellinger", "PDF_KS",
     "PDF_Lognormal", "Dendrogram_Hist", "Dendrogram_Num"]
