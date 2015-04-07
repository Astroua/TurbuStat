
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


def viz_dist_mat(df, new_index, show_img=True):
    '''
    Re-order a triangular data frame.
    '''
    from pandas import DataFrame

    sym_dist = df.values.T + df.values

    sym_df = DataFrame(sym_dist, index=df.index, columns=df.columns)

    reorder_df = sym_df.reindex(index=new_index, columns=new_index)

    # Now restore only the upper triangle

    upptri_df = DataFrame(reorder_df.values * (df.values != 0.0),
                          index=new_index,
                          columns=new_index)

    if show_img:
        import matplotlib.pyplot as p

        p.imshow(upptri_df.values, interpolation='nearest',
                 cmap='binary')
        cbar = p.colorbar()
        cbar.set_label('Distance', fontsize=20)
        p.show()
    return upptri_df


def drop_nans(csv1, csv2):
    '''
    Drop common NaNned rows and columns
    '''

    dist1 = read_csv(csv1, index_col=0)
    dist2 = read_csv(csv2, index_col=0)

    # Replace 0 with nans
    dist1 = dist1.replace(0.0, np.NaN)
    dist2 = dist2.replace(0.0, np.NaN)

    nan_col_1 = ~np.isnan(dist1.mean(0))
    nan_col_2 = ~np.isnan(dist2.mean(0))

    nan_col = nan_col_1 * nan_col_2

    # First column is all zeros, keep it
    nan_col[0] = True

    dist1 = dist1[nan_col]
    dist2 = dist2[nan_col]

    dist1 = dist1.T[nan_col]
    dist2 = dist2.T[nan_col]

    dist1 = dist1.replace(np.NaN, 0.0)
    dist2 = dist2.replace(np.NaN, 0.0)

    return dist1.values, dist2.values

if __name__ == "__main__":

    # Choose which test is to be run.
    run_mantel = True
    run_procrust = True

    # Input folder where tables are saved.
    folder = str(sys.argv[1])

    csv_files = glob.glob(folder+'*.csv')

    # Now we want to sort by Design/Fiducial, Face and Statistics
    designs = np.arange(0, 32)
    fiducials = np.arange(0, 5)

    statistics = ['Cramer', 'PCA']  # , 'PDF', 'SCF', 'VCA', 'VCS',
                  # 'VCS_Break', 'VCS_Density', 'VCS_Velocity']

    stats_dict = dict.fromkeys(statistics)

    timesteps = list(np.arange(21, 31))

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
            mantel_output = np.zeros((2, 10, 10))

            csvs = stats_dict[stat]
            for (i, j) in combinations(timesteps, 2):
                dist1, dist2 = drop_nans(csvs[i], csvs[j])

                # Symmeterize
                dist1 += dist1.T
                dist2 += dist2.T
                output = mantel_test(dist1, dist2, nperm=100)

                mantel_output[:, timesteps.index(i), timesteps.index(j)] = output

                mantel_results[stat] = \
                    {'value': mantel_output[0, :, :],
                     'pvals': mantel_output[1, :, :]}

        for stat in statistics:

            vals = mantel_results[stat]['value']
            vals[vals == 0.0] = np.NaN
            # p.subplot(121)
            # p.title(stat.replace("_", " "))
            p.imshow(vals,# vmax=1, vmin=-1,
                     interpolation='nearest', cmap='binary')
            p.xticks(np.arange(0, 10), timesteps)
            p.yticks(np.arange(0, 10), timesteps)
            p.xlabel('Timesteps')
            p.ylabel('Timesteps')
            cb = p.colorbar()
            cb.set_label('Correlation')
            # p.subplot(122)
            # p.imshow(mantel_results[stat]['pvals'],# vmax=1, vmin=0,
                     # interpolation='nearest', cmap='gray')
            # p.colorbar()
            p.tight_layout()
            p.show()

    # Now run Procrustes analysis.
    if run_procrust:
        execfile("/Users/eric/Dropbox/code_development/advanced_turbustat/procrustes.py")

        seed = 3248030

        procrustes_results = dict.fromkeys(statistics)

        for stat in statistics:
            procrustes_output = np.zeros((2, 10, 10))

            csvs = stats_dict[stat]
            for (i, j) in combinations(timesteps, 2):
                dist1, dist2 = drop_nans(csvs[i], csvs[j])

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

                output = procrustes_analysis(pos_1, pos_2, nperm=100)

                procrustes_output[:, timesteps.index(i), timesteps.index(j)] = output

                procrustes_results[stat] = \
                    {'value': procrustes_output[0, :, :],
                     'pvals': procrustes_output[1, :, :]}

        for stat in statistics:
            # p.subplot(121)
            vals = procrustes_results[stat]['value']
            vals[vals == 0.0] = np.NaN
            # p.subplot(121)
            # p.title(stat.replace("_", " "))
            p.imshow(vals,# vmax=1, vmin=-1,
                     interpolation='nearest', cmap='gray')
            p.xticks(np.arange(0, 10), timesteps)
            p.yticks(np.arange(0, 10), timesteps)
            p.xlabel('Timesteps')
            p.ylabel('Timesteps')
            cb = p.colorbar()
            cb.set_label('Sum of Residuals')
            # p.subplot(122)
            # p.imshow(procrustes_results[stat]['pvals'],# vmax=1, vmin=0,
            #          interpolation='nearest', cmap='gray')
            # p.colorbar()
            p.tight_layout()
            p.show()
