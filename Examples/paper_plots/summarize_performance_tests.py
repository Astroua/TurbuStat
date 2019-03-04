
'''
Make representative plots for the appendix and formalize the csv files
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

font_scale = 1.25

width = 8.4
# Keep the default ratio used in seaborn. This can get overwritten.
height = (4.4 / 6.4) * width

figsize = (width, height)

sb.set(font='Times New Roman', style='ticks',
       rc={'text.usetex': True})
sb.set_context("paper", font_scale=font_scale,
               rc={"figure.figsize": figsize})
sb.set_palette("colorblind")

plt.rcParams['axes.unicode_minus'] = False

tab_2D = pd.read_csv("../twoD_scaling_tests.csv", index_col=0)
tab_3D = pd.read_csv("../threeD_scaling_tests.csv", index_col=0)

mask = slice(3, None)

xvals = tab_2D.index[mask]

symbols = ['s', 'o', 'p', 'P', 'X']

fig = plt.figure(figsize=figsize)

val_range = np.logspace(np.log10(128), np.log10(2100))

# 2D memory
ax_2D_mem = plt.subplot(2, 2, 1)
ax_2D_mem.semilogy(xvals, tab_2D['Wavelet_memory'][mask], symbols[0],
                   linestyle='--', label='Wavelet')
ax_2D_mem.semilogy(xvals, tab_2D['DeltaVariance_memory'][mask], symbols[1],
                   linestyle='--', label='Delta-Variance')
ax_2D_mem.semilogy(xvals, tab_2D['Genus_memory'][mask], symbols[2],
                   linestyle='--', label='Genus')
ax_2D_mem.semilogy(xvals, tab_2D['Dendrogram_Stats_memory'][mask], symbols[3],
                   linestyle='--', label='Dendrograms')
ax_2D_mem.semilogy(xvals, tab_2D['PowerSpectrum_memory'][mask], symbols[4],
                   linestyle='--', label='Power-Spectrum')
ax_2D_mem.semilogy(val_range, (val_range / 256.)**2,
                   linestyle='-', linewidth=5, color='k',
                   alpha=0.3, zorder=-1)
ax_2D_mem.text(2150., 70, r"$N^2$", color='k', alpha=0.5)

ax_2D_mem.semilogy(val_range, (val_range / 256.)**3,
                   linestyle=':', linewidth=5, color='k',
                   alpha=0.3, zorder=-1)
ax_2D_mem.text(2150., 450., r"$N^3$", color='k', alpha=0.5)
ax_2D_mem.set_ylabel('Memory (MB)')
ax_2D_mem.grid()
ax_2D_mem.legend(frameon=True, ncol=2, loc='lower center', fontsize=10)
ax_2D_mem.set_ylim([10**-3, 5 * 10**5])
ax_2D_mem.set_xlim([100, 2350])
plt.setp(ax_2D_mem.get_xticklabels(), visible=False)

ax_2D_time = plt.subplot(2, 2, 3, sharex=ax_2D_mem)
ax_2D_time.semilogy(xvals, tab_2D['Dendrogram_Stats_time'][mask], symbols[0],
                    linestyle='--', label='Dendrograms')
ax_2D_time.semilogy(xvals, tab_2D['Genus_time'][mask], symbols[1],
                    linestyle='--', label='Genus')
ax_2D_time.semilogy(xvals, tab_2D['DeltaVariance_time'][mask], symbols[2],
                    linestyle='--', label='Delta-Variance')
ax_2D_time.semilogy(xvals, tab_2D['Bispectrum_time'][mask], symbols[3],
                    linestyle='--', label='Bispectrum')
ax_2D_time.semilogy(xvals, tab_2D['PowerSpectrum_time'][mask], symbols[4],
                    linestyle='--', label='Power-Spectrum')
ax_2D_time.semilogy(val_range, (val_range / 256.)**2,
                    linestyle='-', linewidth=5, color='k',
                    alpha=0.3, zorder=-1)
ax_2D_time.text(2150., 70, r"$N^2$", color='k', alpha=0.5)

ax_2D_time.semilogy(val_range, (val_range / 256.)**3,
                    linestyle=':', linewidth=5, color='k',
                    alpha=0.3, zorder=-1)
ax_2D_time.text(2150., 450., r"$N^3$", color='k', alpha=0.5)

ax_2D_time.set_ylabel('Time (s)')
ax_2D_time.set_xlabel('Size (pix)')
ax_2D_time.grid()
ax_2D_time.legend(frameon=True, ncol=2, loc='upper center', fontsize=10)

ax_2D_time.set_ylim([10**0, 8 * 10**6])


ax_3D_mem = plt.subplot(2, 2, 2, sharex=ax_2D_mem, sharey=ax_2D_mem)
ax_3D_mem.semilogy(xvals, tab_3D['SCF_memory'][mask], symbols[0],
                   linestyle='--', label='SCF')
ax_3D_mem.semilogy(xvals, tab_3D['VCS_memory'][mask], symbols[1],
                   linestyle='--', label='VCS')
ax_3D_mem.semilogy(xvals, tab_3D['VCA_memory'][mask], symbols[2],
                   linestyle='--', label='VCA')
ax_3D_mem.semilogy(xvals, tab_3D['Cramer_Distance_memory'][mask], symbols[3],
                   linestyle='--', label='Cramer')
ax_3D_mem.semilogy(xvals, tab_3D['PCA_memory'][mask], symbols[4],
                   linestyle='--', label='PCA')
ax_3D_mem.semilogy(val_range, (val_range / 256.)**2,
                   linestyle='-', linewidth=5, color='k',
                   alpha=0.3, zorder=-1)
ax_3D_mem.text(2150., 70, r"$N^2$", color='k', alpha=0.5)

ax_3D_mem.semilogy(val_range, (val_range / 256.)**3,
                   linestyle=':', linewidth=5, color='k',
                   alpha=0.3, zorder=-1)
ax_3D_mem.text(2150., 450., r"$N^3$", color='k', alpha=0.5)
ax_3D_mem.grid()
ax_3D_mem.legend(frameon=True, ncol=2, fontsize=10)
plt.setp(ax_3D_mem.get_xticklabels(), visible=False)
plt.setp(ax_3D_mem.get_yticklabels(), visible=False)

ax_3D_time = plt.subplot(2, 2, 4, sharex=ax_2D_mem, sharey=ax_2D_time)
ax_3D_time.semilogy(xvals, tab_3D['PCA_time'][mask], symbols[0],
                    linestyle='--', label='PCA')
ax_3D_time.semilogy(xvals, tab_3D['SCF_time'][mask], symbols[1],
                    linestyle='--', label='SCF')
ax_3D_time.semilogy(xvals, tab_3D['VCA_time'][mask], symbols[2],
                    linestyle='--', label='VCA')
ax_3D_time.semilogy(xvals, tab_3D['Cramer_Distance_time'][mask], symbols[3],
                    linestyle='--', label='Cramer')
ax_3D_time.semilogy(xvals, tab_3D['VCS_time'][mask], symbols[4],
                    linestyle='--', label='VCS')
ax_3D_time.semilogy(val_range, (val_range / 256.)**2,
                    linestyle='-', linewidth=5, color='k',
                    alpha=0.3, zorder=-1)
ax_3D_time.text(2150., 70, r"$N^2$", color='k', alpha=0.5)

ax_3D_time.semilogy(val_range, (val_range / 256.)**3,
                    linestyle=':', linewidth=5, color='k',
                    alpha=0.3, zorder=-1)
ax_3D_time.text(2150., 450., r"$N^3$", color='k', alpha=0.5)

ax_3D_time.set_xlabel('Size (pix)')
ax_3D_time.set_xticks([256, 512, 1024, 2048])
ax_3D_time.grid()
ax_3D_time.legend(frameon=True, ncol=2, loc='upper left', fontsize=10)
plt.setp(ax_3D_time.get_yticklabels(), visible=False)

plt.subplots_adjust(hspace=0.03, wspace=0.03)

plt.savefig("../figures/scaling_tests.png")
plt.savefig("../figures/scaling_tests.pdf")
plt.close()
