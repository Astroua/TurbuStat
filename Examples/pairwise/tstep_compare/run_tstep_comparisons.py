
import numpy as np
from sklearn.manifold import MDS
from pandas import read_csv
import sys
import glob
import numpy as np
from itertools import combinations
import matplotlib.pyplot as p
import seaborn as sb

sb.set_context('talk')
sb.set_style('ticks')

'''
Run Mantel Test and Procrustes analysis on the pairwise comparisons.
'''


def viz_dist_mat(csv):
    pass

if __name__ == "__main__":


    # Choose which test is to be run.
    run_mantel = True
    run_procrust = False

    # Input folder where tables are saved.
    folder = str(sys.argv[1])

    csv_files = glob.glob(folder+'*.csv')

    # Now we want to sort by Design/Fiducial, Face and Statistics
    designs = np.arange(0, 32)
    fiducials = np.arange(0, 5)

    statistics = ['Cramer', 'PCA']  # , 'PDF', 'SCF', 'VCA', 'VCS',
                  # 'VCS_Break', 'VCS_Density', 'VCS_Velocity']

    stats_dict = dict.fromkeys(statistics)

    timesteps = list(np.arange(21, 25))

    for stat in statistics:
        stats_dict[stat] = dict.fromkeys(timesteps)
        for csv in csv_files:
            if "_"+stat+"_distmat" not in csv:
                continue

            for tstep in timesteps:
                if "_"+str(tstep)+"_" in csv:
                    stats_dict[stat][tstep] = csv
                    break

            else:
                print "Can't classify %s" % (csv)

    # Now load in the Mantel Test
    if run_mantel:
        from turbustat.statistics.mantel import mantel_test

        mantel_results = dict.fromkeys(statistics)

        for stat in statistics:
            mantel_output = np.zeros((2, 4, 4))

            csvs = stats_dict[stat]
            for (i, j) in combinations(timesteps, 2):
                dist1 = read_csv(csvs[i], index_col=0).values
                dist2 = read_csv(csvs[j], index_col=0).values

                # Symmeterize
                dist1 += dist1.T
                dist2 += dist2.T
                output = mantel_test(dist1, dist2, nperm=1000)

                mantel_output[:, timesteps.index(i), timesteps.index(j)] = output

                mantel_results[stat] = \
                    {'value': mantel_output[0, :, :],
                     'pvals': mantel_output[1, :, :]}

        for stat in statistics:
            p.subplot(121)
            p.title(stat.replace("_", " "))
            p.imshow(mantel_results[stat]['value'],# vmax=1, vmin=-1,
                     interpolation='nearest', cmap='seismic')
            p.colorbar()
            p.subplot(122)
            p.imshow(mantel_results[stat]['pvals'],# vmax=1, vmin=0,
                     interpolation='nearest', cmap='gray')
            p.colorbar()
            p.show()

    # Now run Procrustes analysis.
    if run_procrust:
        execfile("/Users/eric/Dropbox/code_development/advanced_turbustat/procrustes.py")

        seed = 3248030

        procrustes_results = dict.fromkeys(statistics)

        for stat in statistics:
            procrustes_output = np.zeros((2, 4, 4))

            csvs = stats_dict[stat]
            for (i, j) in combinations(timesteps, 2):
                dist1 = read_csv(csvs[i], index_col=0).values
                dist2 = read_csv(csvs[j], index_col=0).values

                # Symmeterize
                dist1 += dist1.T
                dist2 += dist2.T

                # Run MDS and map to lower dim. Try 2 for visualizing.
                mds_1 = MDS(n_components=2, max_iter=3000, eps=1e-9,
                            random_state=seed, dissimilarity="precomputed",
                            n_jobs=1)
                pos_1 = mds_1.fit(dist1).embedding_

                mds_2 = MDS(n_components=2, max_iter=3000, eps=1e-9,
                            random_state=seed, dissimilarity="precomputed",
                            n_jobs=1)
                pos_2 = mds_2.fit(dist2).embedding_

                output = procrustes_analysis(pos_1, pos_2, nperm=1000)

                procrustes_output[:, timesteps.index(i), timesteps.index(j)] = output

                procrustes_results[stat] = \
                    {'value': procrustes_output[0, :, :],
                     'pvals': procrustes_output[1, :, :]}

        for stat in statistics:
            p.subplot(121)
            p.title(stat.replace("_", " "))
            p.imshow(procrustes_results[stat]['value'],# vmax=1, vmin=0,
                     interpolation='nearest', cmap='seismic')
            p.colorbar()
            p.subplot(122)
            p.imshow(procrustes_results[stat]['pvals'],# vmax=1, vmin=0,
                     interpolation='nearest', cmap='gray')
            p.colorbar()
            p.show()
