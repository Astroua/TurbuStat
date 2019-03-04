

'''
Compare Turbustat's Delta-variance to the original IDL code.
'''

from turbustat.statistics import DeltaVariance
from turbustat.simulator import make_extended
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt
import astropy.units as u
import seaborn as sb

font_scale = 1.25

width = 4.2
# Keep the default ratio used in seaborn. This can get overwritten.
height = (4.4 / 6.4) * width
figsize = (width, height)

sb.set_context("paper", font_scale=font_scale,
               rc={"figure.figsize": figsize})
sb.set_palette("colorblind")

col_pal = sb.color_palette()

plt.rcParams['axes.unicode_minus'] = False

size = 256

markers = ['D', 'o']

# Make a single figure example to save space in the paper.

fig = plt.figure(figsize=figsize)

slope = 3.0

test_img = fits.PrimaryHDU(make_extended(size, powerlaw=slope))
# The power-law behaviour continues up to ~1/4 of the size
delvar = DeltaVariance(test_img).run(xlow=3 * u.pix,
                                     xhigh=0.25 * size * u.pix,
                                     boundary='wrap')

plt.xscale("log")
plt.yscale("log")
plt.errorbar(delvar.lags.value, delvar.delta_var,
             yerr=delvar.delta_var_error,
             fmt=markers[0], label='TurbuStat')

# Now plot the IDL output
tab = Table.read("deltavar_{}.txt".format(slope), format='ascii')
# First is pixel scale, second is delvar, then delvar error, and finally
# the fit values
plt.errorbar(tab['col1'], tab['col2'], yerr=tab['col3'],
             fmt=markers[1], label='IDL')

plt.grid()

plt.legend(frameon=True)

plt.ylabel(r"$\Delta$-Variance")
plt.xlabel("Scales (pix)")

plt.tight_layout()

plt.savefig("../figures/delvar_vs_idl.png")
plt.savefig("../figures/delvar_vs_idl.pdf")
plt.close()
