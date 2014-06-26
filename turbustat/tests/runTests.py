from .test_bispec import testBispec
from .test_cramer import testCramer
from .test_delvar import testDelVar
from .test_genus import testGenus
from .test_kurtosis_and_skewness import testKurtSkew
from .test_mvc import testMVC
from .test_pca import testPCA
from .test_pspec import testPSpec
from .test_scf import testSCF
from .test_tsallis import testTsallis
from .test_vca import testVCA
from .test_vcs import testVCS
from .test_wavelet import testWavelet

testBispec().test_Bispec_method()
testBispec().test_Bispec_distance()

testCramer().test_Cramer_method()
testCramer().test_Cramer_distance()

testDelVar().test_DelVar_method()
testDelVar().test_DelVar_distance()

testGenus().test_Genus_method()
testGenus().test_Genus_distance()

testKurtSkew().test_Kurtosis_method()
testKurtSkew().test_Kurtosis_distance()

testKurtSkew().test_Skewness_method()
testKurtSkew().test_Skewness_distance()

testMVC().test_MVC_method()
testMVC().test_MVC_distance()

testPCA().test_PCA_method()
testPCA().test_PCA_distance()

testPSpec().test_PSpec_method()
testPSpec().test_PSpec_distance()

testSCF().test_SCF_method()
testSCF().test_SCF_distance()

testTsallis().test_Tsallis_method()
testTsallis().test_Tsallis_distnace()

testVCA().test_VCA_method()
testVCA().test_VCA_distance()

testVCS().test_VCS_method()
testVCS().test_VCS_distance()

testWavelet().test_Wavelet_method()
testWavelet().test_Wavelet_distance()