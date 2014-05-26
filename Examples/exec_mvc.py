
'''

Execute MVC

'''

execfile("mvc.py")

from astropy.io.fits import getdata
import numpy as np
import matplotlib.pyplot as p

linewidth, hdr_lw = getdata("../../../../simcloud_stats/run_128_12_5_0_20_1/run_128_12_5_0_20_1_21/run_128_12_5_0_20_1_21_face2.linewidth.fits",header=True)
linewidth_err, hdr_lwerr = getdata("../../../../simcloud_stats/run_128_12_5_0_20_1/run_128_12_5_0_20_1_21/run_128_12_5_0_20_1_21_face2.linewidth.error.fits",header=True)
centroid, hdr_cent = getdata("../../../../simcloud_stats/run_128_12_5_0_20_1/run_128_12_5_0_20_1_21/run_128_12_5_0_20_1_21_face2.centroid.fits",header=True)
centroid_err, hdr_centerr = getdata("../../../../simcloud_stats/run_128_12_5_0_20_1/run_128_12_5_0_20_1_21/run_128_12_5_0_20_1_21_face2.centroid.error.fits",header=True)
int_intensity, hdr_lw = getdata("../../../../simcloud_stats/run_128_12_5_0_20_1/run_128_12_5_0_20_1_21/run_128_12_5_0_20_1_21_face2.moment0.fits",header=True)
int_intensity_err, hdr_lwerr = getdata("../../../../simcloud_stats/run_128_12_5_0_20_1/run_128_12_5_0_20_1_21/run_128_12_5_0_20_1_21_face2.moment0.error.fits",header=True)

# linewidth, hdr_lw = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.linewidth.fits",header=True)
# linewidth_err, hdr_lwerr = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.linewidth.error.fits",header=True)
# centroid, hdr_cent = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.centroid.fits",header=True)
# centroid_err, hdr_centerr = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.centroid.error.fits",header=True)
# int_intensity, hdr_lw = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.moment0.fits",header=True)
# int_intensity_err, hdr_lwerr = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.moment0.error.fits",header=True)


distances = np.arange(1,20,1)

test = MVC(centroid * centroid_err**2., int_intensity * int_intensity_err**2., linewidth * linewidth_err**2., distances=None)

test.run()

p.subplot(121)
p.loglog(test.freq, test.ps1D,"kD")
p.xlabel("Wave Vector (k)")
p.ylabel("P(k)")
p.grid(True)
p.subplot(122)
p.imshow(np.log10(test.ps2D), origin="lower", interpolation="nearest")
p.colorbar()
p.show()

