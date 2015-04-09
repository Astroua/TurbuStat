
import numpy as np
from sklearn.manifold import MDS
from pandas import read_csv, DataFrame
import sys
import glob
import numpy as np
from itertools import combinations, izip
import matplotlib.pyplot as p

'''
Run Mantel Test and Procrustes analysis on the pairwise comparisons.
'''

# Choose which test is to be run.
run_mantel = False
run_procrust = True

# Input folder where tables are saved.
folder = str(sys.argv[1])

csv_files = glob.glob(folder+'*.csv')

# Now we want to sort by Design/Fiducial, Face and Statistics
designs = np.arange(0, 32)
fiducials = np.arange(0, 5)

statistics = ['Cramer', 'PCA', 'PDF', 'SCF', 'VCA', 'VCS',
              'VCS_Break', 'VCS_Density', 'VCS_Velocity']

stats_dict = dict.fromkeys(statistics)

for stat in statistics:
    stats_dict[stat] = {"Fiducials": None, 'Designs': None}
    stats_dict[stat]['Fiducials'] = {0: [], 1: [], 2: []}
    stats_dict[stat]['Designs'] = {0: [], 1: [], 2: []}
    for csv in csv_files:
        if "_"+stat+"_distmat" not in csv:
            continue

        if 'Design' in csv:
            for face in [0, 1, 2]:
                if '_'+str(face)+'_' in csv:
                    stats_dict[stat]['Designs'][face].append(csv)
                    break
        elif 'Fiducial' in csv:
            for face in [0, 1, 2]:
                if '_'+str(face)+'_' in csv:
                    stats_dict[stat]['Fiducials'][face].append(csv)
                    break

        else:
            print "Can't classify %s" % (csv)

        # csv_files.remove(csv)

# Now load in the Mantel Test
if run_mantel:
    from turbustat.statistics.mantel import mantel_test

    # First compare the fiducials.

    mantel_fiducials = dict.fromkeys(statistics)
    mantel_designs = dict.fromkeys(statistics)

    for stat in statistics:
        mantel_results_fid = np.zeros((2, 3, 5, 5))
        mantel_results_des = np.zeros((2, 3, 5, 32))

        mantel_fiducials[stat] = dict.fromkeys([0, 1, 2])
        mantel_designs[stat] = dict.fromkeys([0, 1, 2])

        for face in [0, 1, 2]:
            csvs = stats_dict[stat]['Fiducials'][face]
            for (i, j) in combinations(fiducials, 2):
                dist1 = read_csv(csvs[i], index_col=0).values
                dist2 = read_csv(csvs[j], index_col=0).values

                # Symmeterize
                dist1 += dist1.T
                dist2 += dist2.T
                output = mantel_test(dist1, dist2, nperm=1000)

                mantel_results_fid[:, face, i, j] = output

            mantel_fiducials[stat][face] = \
                {'value': mantel_results_fid[0, face, :, :],
                 'pvals': mantel_results_fid[1, face, :, :]}


            # Now compare fiducials to designs.
            csv_designs = stats_dict[stat]['Designs'][face]

            for csv_fid in csvs:
                fid_num = int(csv_fid.split('/')[-1][18]) - 1
                for csv_des in csv_designs:
                        des_num = csv_des.split('/')[-1][16:18]
                        if des_num[-1] == "_":
                            des_num = des_num[0]
                        des_num = int(des_num) - 1

                        dist1 = read_csv(csv_fid, index_col=0).values
                        dist2 = read_csv(csv_des, index_col=0).values

                        # Some of the Designs don't have 10 timesteps.
                        # Cut fiducial down in this case.
                        if dist1.shape != dist2.shape:
                            new_size = dist2.shape[0]
                            dist1 = dist1[:new_size, :new_size]

                        # Symmeterize
                        dist1 += dist1.T
                        dist2 += dist2.T
                        output = mantel_test(dist1, dist2, nperm=1000)

                        mantel_results_des[:, face, fid_num, des_num] = output

            mantel_designs[stat][face] = {'value': mantel_results_des[0, face, :, :],
                                          'pvals': mantel_results_des[1, face, :, :]}

    for stat in statistics:
        p.subplot(121)
        p.title(stat.replace("_", " "))
        p.imshow(mantel_fiducials[stat][0]['value'], vmax=1, vmin=-1,
                 interpolation='nearest', cmap='seismic')
        p.colorbar()
        p.subplot(122)
        p.imshow(mantel_fiducials[stat][0]['pvals'], vmax=1, vmin=0,
                 interpolation='nearest', cmap='gray')
        p.colorbar()
        p.show()

        p.subplot(121)
        p.title(stat.replace("_", " "))
        p.imshow(mantel_designs[stat][0]['value'].T, vmax=1, vmin=-1,
                 interpolation='nearest', cmap='seismic')
        p.colorbar()
        p.subplot(122)
        p.imshow(mantel_designs[stat][0]['pvals'].T, vmax=1, vmin=0,
                 interpolation='nearest', cmap='gray')
        p.colorbar()
        p.show()


# Now run Procrustes analysis.
if run_procrust:
    execfile("/Users/eric/Dropbox/code_development/advanced_turbustat/procrustes.py")

    seed = 3248030

    proc_fiducials = dict.fromkeys(statistics)
    proc_designs = dict.fromkeys(statistics)

    for stat in statistics:
        proc_results = np.zeros((2, 3, 5, 5))
        proc_results_des = np.zeros((2, 3, 5, 32))

        proc_fiducials[stat] = dict.fromkeys([0, 1, 2])
        proc_designs[stat] = dict.fromkeys([0, 1, 2])

        for face in [0, 1, 2]:
            csvs = stats_dict[stat]['Fiducials'][face]
            for (i, j) in combinations(fiducials, 2):
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

                output = procrustes_analysis(pos_1, pos_2, nperm=10)

                proc_results[:, face, i, j] = output

            proc_fiducials[stat][face] = {'value': proc_results[0, face, :, :],
                                          'pvals': proc_results[1, face, :, :]}

        # Now compare fiducials to designs.
            csv_designs = stats_dict[stat]['Designs'][face]

            for csv_fid in csvs:
                fid_num = int(csv_fid.split('/')[-1][18]) - 1
                for csv_des in csv_designs:
                        des_num = csv_des.split('/')[-1][16:18]
                        if des_num[-1] == "_":
                            des_num = des_num[0]
                        des_num = int(des_num) - 1

                        dist1 = read_csv(csv_fid, index_col=0).values
                        dist2 = read_csv(csv_des, index_col=0).values

                        # Some of the Designs don't have 10 timesteps.
                        # Cut fiducial down in this case.
                        if dist1.shape != dist2.shape:
                            new_size = dist2.shape[0]
                            dist1 = dist1[:new_size, :new_size]

                        # Symmeterize
                        dist1 += dist1.T
                        dist2 += dist2.T
                        output = procrustes_analysis(dist1, dist2, nperm=10)

                        proc_results_des[:, face, fid_num, des_num] = output

            proc_designs[stat][face] = {'value': proc_results_des[0, face, :, :],
                                          'pvals': proc_results_des[1, face, :, :]}

    for stat in statistics:
        p.subplot(121)
        p.title(stat.replace("_", " "))
        p.imshow(proc_fiducials[stat][0]['value'], vmax=1, vmin=0,
                 interpolation='nearest', cmap='seismic')
        p.colorbar()
        p.subplot(122)
        p.imshow(proc_fiducials[stat][0]['pvals'], vmax=1, vmin=0,
                 interpolation='nearest', cmap='gray')
        p.colorbar()
        p.show()

        p.subplot(121)
        p.title(stat.replace("_", " "))
        p.imshow(proc_designs[stat][0]['value'].T, vmax=1, vmin=0,
                 interpolation='nearest', cmap='seismic')
        p.colorbar()
        p.subplot(122)
        p.imshow(proc_designs[stat][0]['pvals'].T, vmax=1, vmin=0,
                 interpolation='nearest', cmap='gray')
        p.colorbar()
        p.show()
