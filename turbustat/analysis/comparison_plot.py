# Licensed under an MIT open source license - see LICENSE

import numpy as np
import os
import warnings
from itertools import repeat
import matplotlib as mpl
import matplotlib.pyplot as p
from pandas import read_csv, DataFrame

from ..statistics import statistics_list
from .analysis_utils import parameters_dict


def comparison_plot(path, num_fids=5, verbose=False,
                    obs_to_fid=False, obs_to_fid_shade=True, legend=True,
                    legend_labels=None,
                    statistics=statistics_list,
                    comparisons=["0_0", "0_1", "0_2", "1_0", "1_1", "1_2",
                                 "2_0", "2_1", "2_2", "0_obs", "1_obs",
                                 "2_obs"],
                    out_path=None, design_matrix=None, sharey=True,
                    obs_legend=False):
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
    obs_to_fid : bool, optional
        Include observational to fiducial distances in the distance subplots.
    obs_to_fid_shade : bool, optional
        Plots the observational distances as a single band instead of by
        fiducial.
    legend : bool, optional
        Toggle having a legend.
    legend_labels : list, optional
        Provide a list of labels to be used in the legend in lieu of
        "Fiducial #". Length must match the number of fiducials!
    statistics : list, optional
        Statistics to plot. Default is all.
    comparisons : list, optional
        The face comparisons to include in the plots. The order is set here as
        well.
    design_matrix : str or pandas.DataFrame, optional
        The experimental design, which is used for the distances. If a string,
        is given, it is assumed that it's the path to the saved csv file. When
        given, the labels of the plots will be coded in 'binary' according to
        the levels in the design.
    sharey : bool, optional
        When enabled, each subplot has the same y limits.
    obs_legend : bool, optional
        Turn on legend for the observational comparisons. When disabled,
        labels are plotted over the shaded region.
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

    data_files["0_obs"] = ["Face 0 to Obs", None, None]
    data_files["1_obs"] = ["Face 1 to Obs", None, None]
    data_files["2_obs"] = ["Face 2 to Obs", None, None]

    # Remove comparisons that aren't requested
    for key in data_files.keys():
        if key not in comparisons:
            del data_files[key]

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
                if "face_"+str(key) in x:
                    obs_to_fid_data[key] = read_csv(os.path.join(path, x),
                                                    index_col=0)
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

    # Check for a design
    if design_matrix is not None:
        if isinstance(design_matrix, DataFrame):
            pass
        elif isinstance(design_matrix, unicode) or isinstance(design_matrix, str):
            design_matrix = read_csv(design_matrix, index_col=0)
        else:
            print type(design_matrix)
            raise TypeError("design_matrix must be a pandas.DataFrame or "
                            "a path to a csv file.")

        # Set -1 to 0 for cleanliness
        design_matrix[design_matrix == -1] = 0.0

        # Start with getting the parameter symbols
        design_labels = \
            [("".join([parameters_dict[col] for col in design_matrix.columns]))]

        for ind in design_matrix.index:

            label = "".join([str(int(val)) for val in design_matrix.ix[ind]])

            design_labels.append(label)

    else:
        design_labels = None

    # Set the colour cycle
    colour_cycle = mpl.rcParams['axes.color_cycle']
    if len(colour_cycle) < num_fids:
        # Need to define our own in this case.
        mpl.rcParams['axes.color_cycle'] = \
            [mpl.cm.jet(i) for i in np.linspace(0.0, 1.0, num_fids)]
    else:
        mpl.rcParams['axes.color_cycle'] = colour_cycle[:num_fids]

    if legend_labels is not None:
        if len(legend_labels) != num_fids:
            raise Warning("The number of labels given in legend_labels does"
                          " not match the number of fiducials specified: "
                          + str(num_fids))

    for stat in statistics:
        # Divide by 2 b/c there should be 2 files for each comparison b/w faces
        (fig, ax) = _plot_size(len(data_files.keys()), sharey=sharey)
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
            try:
                _plotter(axis, data_files[key][2][stat],
                         data_files[key][1][stat],
                         num_fids, data_files[key][0], stat, bottom, left,
                         legend=legend, legend_labels=legend_labels,
                         labels=design_labels)
                enable_continue = False
            except KeyError:
                warnings.warn("Could not find data for "+stat+" in "+key)
                enable_continue = True
                break

            if obs_to_fid:
                obs_key = int(key[0])
                _horiz_obs_plot(axis, obs_to_fid_data[obs_key][stat],
                                num_fids, shading=obs_to_fid_shade,
                                legend=obs_legend)

        if enable_continue:
            continue

        # If the labels are given by the design, we need to adjust the bottom
        # of the subplots
        if design_labels is not None:
            top = 0.95
            bottom = 0.15
            fig.subplots_adjust(top=top, bottom=bottom)

        if verbose:
            # p.autoscale(True)
            fig.show()
        else:
            # p.autoscale(True)
            save_name = "distance_comparisons_" + stat + ".pdf"
            if out_path is not None:
                if out_path[-1] != "/":
                    out_path += "/"
                save_name = out_path + save_name
            fig.savefig(save_name)
            fig.clf()


def _plot_size(num, sharey=True):
    if num <= 3:
        return p.subplots(num, sharex=True, sharey=sharey)
    elif num > 3 and num <= 8:
        rows = num / 2 + num % 2
        return p.subplots(nrows=rows, ncols=2, figsize=(14, 14),
                          dpi=100, sharex=True, sharey=sharey)
    elif num == 9:
        return p.subplots(nrows=3, ncols=3, figsize=(14, 14), dpi=100,
                          sharex=True, sharey=sharey)
    else:
        print "There should be a maximum of 9 comparisons."
        return


def _plotter(ax, data, fid_data, num_fids, title, stat, bottom, left,
             legend=True, legend_labels=None, labels=None, ylims=None):

    num_design = (max(data.shape) / num_fids)

    if labels is not None:
        if len(labels) - 1 != num_design:
            raise Warning("Design matrix contains different number of designs "
                          "than the data. Double check the inputted "
                          "design_matrix.")

    x_vals = np.arange(0, num_design)
    xtick_labels = [str(i) for i in x_vals]
    fid_labels = [str(i) for i in range(num_fids-1)]
    # Plot designs
    for i in range(num_fids):
        y_vals = data.ix[int(i * num_design):int(((i + 1) * num_design)-1)]
        if legend_labels is not None:
            ax.plot(x_vals, y_vals, "-o", label=legend_labels[i], alpha=0.6)
        else:
            ax.plot(x_vals, y_vals, "-o", label="Fiducial " + str(i),
                    alpha=0.6)
    # Set title in upper left hand corner
    ax.set_title(title, fontsize=12)
    # ax.annotate(title, xy=(1, 0), xytext=(0.9, 0.05), va='top',
    #             xycoords='axes fraction', textcoords='axes fraction',
    #             fontsize=12, alpha=0.75)
    if left:
        # Set the ylabel using the stat name. Replace underscores
        ax.set_ylabel(stat.replace("_", " ")+"\nDistance", fontsize=10,
                      multialignment='center')
    else:
        ax.set_ylabel("")

    # If the plot is on the bottom of a column, add labels
    if bottom:
        trans = ax.get_xaxis_transform()

        if labels is not None:
            yposn = -0.28
        else:
            yposn = -0.15

        # Put two 'labels' for the x axis
        ax.annotate("Designs", xy=(num_design/2 - 9, yposn),
                    xytext=(num_design/2 - 1, yposn),
                    va='top', xycoords=trans,
                    fontsize=12)
        fid_x = num_design + num_fids/2 - 2.5
        ax.annotate("Fiducials", xy=(fid_x, yposn),
                    xytext=(fid_x, yposn),
                    va='top', xycoords=trans,
                    fontsize=12)

    # Plot fiducials
    if fid_data is not None:
        x_fid_vals = np.arange(num_design, num_design + num_fids)
        prev = 0
        for i, posn in enumerate(np.arange(num_fids - 1, 0, -1)):
            ax.plot(x_fid_vals[:len(x_fid_vals)-i-1],
                    fid_data[prev:posn+prev], "ko", alpha=0.6,
                    label="_nolegend_")
            prev += posn
    # Make the legend
    if legend:
        ax.legend(loc="upper right", prop={'size': 10})
    if labels is None:
        ax.set_xlim([-1, num_design + num_fids + 8])
        ax.set_xticks(np.append(x_vals, x_fid_vals))
        ax.set_xticklabels(xtick_labels+fid_labels, rotation=90, size=12)
    else:
        ax.set_xlim([-2, num_design + num_fids + 8])
        xticks = np.append([-1], np.append(x_vals, x_fid_vals))
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels+fid_labels, rotation=90, size=12)

    if ylims is not None:
        ax.set_ylim(ylims)


def _horiz_obs_plot(ax, data, num_fids, shading=False, legend=False):
    '''
    Plot a horizontal line with surrounding shading across
    the plot to signify the distance of the observational data.
    '''

    # This eventually needs to be generalized
    labels_dict = {"ophA.13co.fits": "OphA",
                   "ngc1333.13co.fits": "NGC 1333",
                   "ic348.13co.fits": "IC 348"}

    # Also needs to be generalized
    colors = ["g", "r", "b"]

    x_vals = ax.axis()[:2]

    num_obs = data.shape[0] / num_fids

    obs_names = data.index[:num_obs]

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

    fill_betweens = []

    for i, (obs, style) in enumerate(zip(obs_names, linestyles)):

        y_vals = np.asarray(data.ix[i::num_obs])

        yposn = np.nanmean(y_vals)
        ymax = np.nanmax(y_vals)
        ymin = np.nanmin(y_vals)

        if shading:

            # Want to plot a single line at the mean, then shade to show
            # variance.

            ax.fill_between(x_vals, ymax, ymin, facecolor=colors[i],
                            interpolate=True, alpha=0.2,
                            edgecolor=colors[i], linestyle=style)

            if legend:
                rect = p.Rectangle((0, 0), 1, 1, fc=colors[i], linestyle=style,
                                   alpha=0.2)
                fill_betweens.append(rect)

            middle = (ymax + ymin) / 2

            if not legend:
                trans = ax.get_yaxis_transform()
                ax.annotate(labels_dict[obs], xy=(0.9, middle),
                            xytext=(0.9, middle),
                            fontsize=12, xycoords=trans,
                            verticalalignment='center',
                            horizontalalignment='center')

        else:

            for j in range(num_fids):
                y_vals = 2*[data.ix[int(j * num_obs)+i]]
                ax.plot(x_vals, y_vals, "-", label="Fiducial " + str(j),
                        alpha=0.4, linewidth=3)


            trans = ax.get_yaxis_transform()
            ax.annotate(labels_dict[obs], xy=(1.0, ymax), xytext=(1.03, yposn),
                        fontsize=15, xycoords=trans,
                        arrowprops=dict(facecolor='k',
                                        width=0.05, alpha=1.0, headwidth=0.1),
                        horizontalalignment='left',
                        verticalalignment='center')
            ax.annotate(labels_dict[obs], xy=(1.0, ymin), xytext=(1.03, yposn),
                        fontsize=15, xycoords=trans,
                        arrowprops=dict(facecolor='k',
                                        width=0.05, alpha=1.0, headwidth=0.1),
                        horizontalalignment='left',
                        verticalalignment='center')

    if legend:
        ax.legend(fill_betweens, [labels_dict[obs] for obs in obs_names],
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
