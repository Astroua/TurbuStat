# Licensed under an MIT open source license - see LICENSE

'''
Returns a list of all available distance metrics
'''

statistics_list = ["Wavelet", "MVC", "PSpec", "Bispectrum", "DeltaVariance",
                  "Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF", "Cramer",
                  "Skewness", "Kurtosis", "VCS_Density", "VCS_Velocity",
                  "PDF_Hellinger", "PDF_KS", # "PDF_AD",
                  "Dendrogram_Hist", "Dendrogram_Num"]

twoD_statistics_list = \
    ["Wavelet", "MVC", "PSpec", "Bispectrum", "DeltaVariance",
     "Genus", "Tsallis", "Skewness", "Kurtosis",
     "PDF_Hellinger", "PDF_KS", # "PDF_AD",
     "Dendrogram_Hist", "Dendrogram_Num"]