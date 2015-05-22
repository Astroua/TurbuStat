# Licensed under an MIT open source license - see LICENSE

import numpy as np
import os
import matplotlib.pyplot as p
from pandas import read_csv


def comparison_plot(path, num_fids=5, verbose=False, obs_to_des=False,
                    obs_to_fid=False, obs_to_fid_data=None,
                    statistics=["Wavelet", "MVC", "PSpec", "Bispectrum",
                                "DeltaVariance", "Genus", "VCS",
                                "VCS_Density", "VCS_Velocity", "VCA",
                                "Tsallis", "PCA", "SCF", "Cramer",
                                "Skewness", "Kurtosis", "Dendrogram_Hist",
                                "Dendrogram_Num", "PDF"],
                    comparisons=["0_0", "0_1", "0_2", "1_0", "1_1", "1_2",
                                 "2_0", "2_1", "2_2"]):
    '''
    Requires results converted into csv form!!

    This function plots a comparison of the distances between the different
    simulations and the fiducial runs. All face combinations are checked for
    in the given path. The plots adjust to the available amount of data.

    Parameters
    ----------
    path : str
           Path to folder containing the HDF5 files with the distance results.
    analysis_fcn : function, optional
        Function to apply to the time-step data.
    verbose : bool, optional
        Enables plotting.
    cross_compare : bool, optional
        Include comparisons between faces.
    statistics : list, optional
        Statistics to plot. Default is all.
    comparisons : list, optional
        The face comparisons to include in the plots. The order is set here as
        well.
    '''

    if path[-1] != "/":
        path += "/"

    # Set the order by the given comparison list
    order = comparisons

    # All possible face combinations
    data_files = {"0_0": ["Face 0 to 0", None, None],
                  "1_1": ["Face 1 to 1", None, None],
                  "2_2": ["Face 2 to 2", None, None],
                  "0_1": ["Face 0 to 1", None, None],
                  "0_2": ["Face 0 to 2", None, None],
                  "1_2": ["Face 1 to 2", None, None],
                  "2_1": ["Face 2 to 1", None, None],
                  "2_0": ["Face 2 to 0", None, None],
                  "1_0": ["Face 1 to 0", None, None]}

    # Remove comparisons that aren't requested
    for key in data_files.keys():
        if not key in comparisons:
            del data_files[key]

    if obs_to_des:
        data_files["0_obs"] = ["Face 0 to Obs"],
        data_files["1_obs"] = ["Face 1 to Obs"],
        data_files["2_obs"] = ["Face 2 to Obs"],

    if obs_to_fid:
        obs_to_fid_data = {0: None,
                           1: None,
                           2: None}

    # Read in the data and match it to one of the face combinations.
    for x in os.listdir(path):
        if not os.path.isfile(os.path.join(path, x)):
            continue
        if not x[-3:] == "csv":
            continue

        # Separate out the observational csv files.
        if obs_to_fid and 'complete' in x:
            for key in obs_to_fid_data.keys():
                if "face_"+str(key):
                    obs_to_fid_data[key] = read_csv(os.path.join(path, x))
                    break
        else:
            for key in data_files.keys():
                if key in x:
                    data = read_csv(os.path.join(path, x))
                    if "fiducial" in x:
                        data_files[key][1] = data
                    elif "distances" in x:
                        data_files[key][2] = data
                    else:
                        pass
                    break

    # Now delete the keys with no data
    for key in data_files.keys():
        if len(data_files[key]) == 1:
            del data_files[key]
            order.remove(key)

    if data_files.keys() == []:
        print "No csv files found in %s" % (path)
        return

    for stat in statistics:
        # Divide by 2 b/c there should be 2 files for each comparison b/w faces
        (fig, ax) = _plot_size(len(data_files.keys()))
        if len(data_files.keys()) == 1:
            shape = (1, )
            ax = np.array([ax])
        else:
            shape = ax.shape

        if len(shape) == 1:
            ax = ax[:, np.newaxis]
            shape = ax.shape
        for k, (key, axis) in enumerate(zip(order, ax.flatten())):
            bottom = False
            if k >= len(ax.flatten()) - shape[1]:
                bottom = True
            if k / float(shape[0]) in [0, 1, 2]:
                left = True
            _plotter(axis, data_files[key][2][stat], data_files[key][1][stat],
                     num_fids, data_files[key][0], stat, bottom, left)
            if obs_to_fid and k <= k / float(shape[1]):
                obs_key = int(key[0])
                _horiz_obs_plot(axis, obs_to_fit_data[obs_key],
                                num_obs, num_fids)

        if verbose:
            p.autoscale(True)
            fig.show()
        else:
            p.autoscale(True)
            fig.savefig("distance_comparisons_" + stat + ".pdf")
            fig.clf()


def _plot_size(num):
    if num <= 3:
        return p.subplots(num, sharex=True)
    elif num > 3 and num <= 8:
        rows = num / 2 + num % 2
        return p.subplots(nrows=rows, ncols=2, figsize=(14, 14), dpi=100, sharex=True)
    elif num == 9:
        return p.subplots(nrows=3, ncols=3, figsize=(14, 14), dpi=100, sharex=True)
    else:
        print "There should be a maximum of 9 comparisons."
        return


def _plotter(ax, data, fid_data, num_fids, title, stat, bottom, left):

    num_design = (max(data.shape) / num_fids)
    x_vals = np.arange(0, num_design)
    xtick_labels = [str(i) for i in x_vals]
    fid_labels = [str(i) for i in range(num_fids-1)]
    # Plot designs
    for i in range(num_fids):
        y_vals = data.ix[int(i * num_design):int(((i + 1) * num_design)-1)]
        ax.plot(x_vals, y_vals, "-o", label="Fiducial " + str(i), alpha=0.6)
    # Set title in upper left hand corner
    ax.annotate(title, xy=(0, 1), xytext=(12, -6), va='top',
                xycoords='axes fraction', textcoords='offset points',
                fontsize=12, alpha=0.75)
    if left:
        # Set the ylabel using the stat name. Replace underscores
        ax.set_ylabel(stat.replace("_", " ")+"\nDistance", fontsize=10,
                      multialignment='center')
    else:
        ax.set_ylabel("")

    # If the plot is on the bottom of a column, add labels
    if bottom:
        # Put two 'labels' for the x axis
        ax.annotate("Designs", xy=(0.4, -0.25), xytext=(0.4, -0.25),
                    va='top', xycoords='axes fraction',
                    textcoords='offset points',
                    fontsize=12)
        ax.annotate("Fiducials", xy=(0.9, -0.25),
                    xytext=(0.9, -0.25),
                    va='top', xycoords='axes fraction',
                    textcoords='offset points',
                    fontsize=12)

    #Plot fiducials
    # fid_comps = (num_fids**2 + num_fids) / 2
    x_fid_vals = np.arange(num_design, num_design + num_fids)
    prev = 0
    for i, posn in enumerate(np.arange(num_fids - 1, 0, -1)):
        ax.plot(x_fid_vals[:len(x_fid_vals)-i-1],
                fid_data[prev:posn+prev], "ko", alpha=0.6)
        prev += posn
    # Make the legend
    ax.legend(loc="upper right", prop={'size': 10})
    ax.set_xlim([-1, num_design + num_fids + 8])
    ax.set_xticks(np.append(x_vals, x_fid_vals))
    ax.set_xticklabels(xtick_labels+fid_labels, rotation=90, size=12)


def _horiz_obs_plot(ax, data, num_obs, num_fids):
    '''
    Plot a horizontal line with surrounding shading across
    the plot to signify the distance of the observational data.
    '''

    # This eventually needs to be generalized
    labels_dict = {"ophA.13co.fits": "OphA",
                   "ngc1333.13co.fits": "NGC-1333",
                   "oc348.14co.fits": "IC-348"}

    obs_names = data.index()

    x_vals = ax.axis()[:2]

    for i in range(num_fids):
        y_vals = data.ix[int(i * num_obs):int(((i + 1) * num_obs)-1)]
        ax.plot(x_vals, y_vals, "-", label="Fiducial " + str(i), alpha=0.4,
                linewidth=3)

    for i, obs in enumerate(obs_names):

        y_vals = data.ix[i::(num_fids)]

        yposn = np.nanmean(y_vals)

        # Calculate position wrt to axis limit
        y_frac = yposn / float(ax.axis()[-1])

        ax.annotate(labels_dict[obs], xy=(1.1, y_frac), xytext=(1.1, y_frac),
                    va='top', xycoords='axes fraction',
                    textcoords='offset points',
                    fontsize=12)


def timestep_comparisons(path, verbose=False):
    '''
    Use pandas built-in plotting to look at the variation across time-steps.

    Parameters
    ----------
    path : str
        Path to files.
    verbose : bool, optional
        Enables plotting.
    '''
    data_files = [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))
                  and x[-2:] == "h5"]
    if len(data_files) == 0:
        print "The inputed path contains no HDF5 files."
        return None

    data = [HDFStore(filename) for filename in data_files]

    for key in data[0].keys():
        for i, dataset in enumerate(data):
            # p.subplot(3,3,i)
            df = dataset[key].sort(axis=0).sort(axis=1)
            df.T.plot(style="D--")
            p.legend(prop={'size': 8}, loc="best")
            p.title(str(key)[1:])
            locs, xlabels = p.xticks(size=8)
            p.setp(xlabels, rotation=70)

            if verbose:
                p.show()
            else:
                p.savefig(
                    "timestep_comparisons_" + str(data_files[i][38:-19]) + "_" + str(key[1:]) + ".pdf")
                p.close()


def num_fiducials(N):
    '''

    Return the number of fiducials based on the number of lines in the
    comparison file.

    Parameters
    ----------
    N : int
        Number of rows in the data frame.
    '''

    n = 1

    while n < N:
        if n * (n - 1) == 2 * N:
            return n - 1
        else:
            n += 1

    return "Doesn't factor into an integer value."
