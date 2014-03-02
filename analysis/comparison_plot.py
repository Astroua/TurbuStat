
import numpy as np
from pandas import read_hdf, HDFStore
import types, os
import matplotlib.pyplot as p
import matplotlib.cm as cm

def comparison_plot(path, analysis_fcn="mean", verbose=False, \
                    statistics=["Wavelet", "MVC", "PSpec","Bispectrum","DeltaVariance", \
                    "Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF", "Cramer", "Skewness", \
                    "Kurtosis"]):
    '''

    This function plots a comparison of the distances between the different simulations and
    (if available) fiducial runs. Two sets of plots are returned; one for comparisons using
    the analysis function (default is mean over time-steps) and one comparing the values for
    each time-step to each fiducial run. The faces of the cubes (0 or 2) are treated separately.

    Parameters
    **********

    path : str
           Path to folder containing the HDF5 files with the distance results.

    '''
    ## Loop through statistics and load in results for each comparison

    data_files = [os.path.join(path,x) for x in os.listdir(path) if os.path.isfile(os.path.join(path,x)) \
                  and x[-2:]=="h5"]

    ## Sort such that the Fiducials are properly labeled
    for i in np.arange(1,7):
        if i==1:
            sorted_files = [f for f in data_files if f[-25]==str(i)]
        else:
            sorted_files.extend([f for f in data_files if f[-25]==str(i)])
    sorted_files.extend([f for f in data_files if f[-48:-45]=="fid"])

    assert len(data_files) == len(sorted_files)

    data_files = sorted_files

    if len(data_files)==0:
        print "The inputed path contains no HDF5 files."
        return None

    loader = lambda data, x: read_hdf(data,x)

    if isinstance(analysis_fcn, str):
        assert isinstance(getattr(loader(data_files[0], statistics[0]), analysis_fcn), types.UnboundMethodType)

    for i, stat in enumerate(statistics):
        data_face0_0 = []
        fid_data_face0_0 = []

        data_face2_2 = []
        fid_data_face2_2 = []

        data_face2_0 = []
        fid_data_face2_0 = []

        data_face0_2 = []
        fid_data_face0_2 = []

        num_sims = 0
        num_fids = 0

        for data in data_files:
            datum = loader(data, stat).sort(axis=1)
            ## Face 0 to 0
            if data[-23:-20]=="0_0":
                if data.split("/")[-1][:8]=="fiducial":
                    num_fids = num_fiducials(datum.shape[0])
                    assert isinstance(num_fids, int)
                    j_last = 0
                    for j in np.arange(num_fids,0, -1):
                        fid_data_face0_0.append(datum.iloc[j_last:j+j_last,:])
                        j_last = j

                else:
                    data_face0_0.append(datum.sort(axis=0))
                    num_sims = datum.shape[0]
            ## Face 2 to 2
            elif data[-23:-20]=="2_2":
                if data.split("/")[-1][:8]=="fiducial":
                    num_fids = num_fiducials(datum.shape[0])
                    assert isinstance(num_fids, int)
                    j_last = 0
                    for j in np.arange(num_fids,0, -1):
                        fid_data_face2_2.append(datum.iloc[j_last:j+j_last,:])
                        j_last = j
                else:
                    data_face2_2.append(datum.sort(axis=0))
            ## Face 0 to 2
            elif data[-23:-20]=="0_2":
                if data.split("/")[-1][:8]=="fiducial":
                    num_fids = num_fiducials(datum.shape[0])
                    assert isinstance(num_fids, int)
                    j_last = 0
                    for j in np.arange(num_fids,0, -1):
                        fid_data_face2_0.append(datum.iloc[j_last:j+j_last,:])
                        j_last = j
                else:
                    data_face2_0.append(datum.sort(axis=0))
            ## Face 2 to 0
            elif data[-23:-20]=="2_0":
                if data.split("/")[-1][:8]=="fiducial":
                    num_fids = num_fiducials(datum.shape[0])
                    assert isinstance(num_fids, int)
                    j_last = 0
                    for j in np.arange(num_fids,0, -1):
                        fid_data_face0_2.append(datum.iloc[j_last:j+j_last,:])
                        j_last = j
                else:
                    data_face0_2.append(datum.sort(axis=0))
            else:
                print "Check filename for position of face label. Currently works only for \
                            the default output of output.py"
                break

        ## Comparison across simulations
        labels = ["Fiducial "+str(num) for num in range(1,num_fids+2)]
        xtick_labels = ["Design "+str(num) for num in range(1,num_sims+1)]
        xtick_labels = xtick_labels + labels[:-1]

        colours = cm.rainbow(np.linspace(0, 1, num_fids+1))
        ax1 = p.subplot(2,2,1)

        for i, (df, col) in enumerate(zip(data_face0_0,colours)):
            p.scatter(np.arange(1, df.shape[0]+1), getattr(df, analysis_fcn)(axis=1), color=col, label=labels[i])
        p.legend(prop={'size':8}) # make now to avoid repeats
        for i, (df, col)in enumerate(zip(fid_data_face0_0, colours)):
            p.scatter(np.arange(num_sims+1, num_sims+(num_fids+1-i)), getattr(df, analysis_fcn)(axis=1),\
                     color=col, label=labels[i+1])
        p.title("Face 0 to 0")
        p.xlim(0, len(xtick_labels)+3)
        locs, xlabels = p.xticks(np.arange(1, len(xtick_labels)+1), xtick_labels, rotation="vertical", size=8)
        p.setp(xlabels, rotation=70)
        p.ylabel(stat+" Distance")

        ax2 = p.subplot(2,2,2)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        for i, (df, col) in enumerate(zip(data_face2_2,colours)):
            p.scatter(np.arange(1, df.shape[0]+1), getattr(df, analysis_fcn)(axis=1), color=col, label=labels[i])
        p.legend(prop={'size':8}) # make now to avoid repeats
        for i, (df, col)in enumerate(zip(fid_data_face2_2, colours)):
            p.scatter(np.arange(num_sims+1, num_sims+(num_fids+1-i)), getattr(df, analysis_fcn)(axis=1),\
                     color=col, label=labels[i+1])
        p.title("Face 2 to 2")
        p.xlim(0, len(xtick_labels)+3)
        locs, xlabels = p.xticks(np.arange(1, len(xtick_labels)+1), xtick_labels, size=8)
        p.setp(xlabels, rotation=70)
        p.ylabel(stat+" Distance")

        ax3 = p.subplot(2,2,3)
        for i, (df, col) in enumerate(zip(data_face0_2,colours)):
            p.scatter(np.arange(1, df.shape[0]+1), getattr(df, analysis_fcn)(axis=1), color=col, label=labels[i])
        p.legend(prop={'size':8}) # make now to avoid repeats
        for i, (df, col)in enumerate(zip(fid_data_face0_2, colours)):
            p.scatter(np.arange(num_sims+1, num_sims+(num_fids+1-i)), getattr(df, analysis_fcn)(axis=1),\
                     color=col, label=labels[i+1])
        p.title("Face 0 to 2")
        p.xlim(0, len(xtick_labels)+3)
        locs, xlabels = p.xticks(np.arange(1, len(xtick_labels)+1), xtick_labels, size=8)
        p.setp(xlabels, rotation=70)
        p.ylabel(stat+" Distance")

        ax4 = p.subplot(2,2,4)
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        for i, (df, col) in enumerate(zip(data_face2_0,colours)):
            p.scatter(np.arange(1, df.shape[0]+1), getattr(df, analysis_fcn)(axis=1), color=col, label=labels[i])
        p.legend(prop={'size':8}) # make now to avoid repeats
        for i, (df, col)in enumerate(zip(fid_data_face2_0, colours)):
            p.scatter(np.arange(num_sims+1, num_sims+(num_fids+1-i)), getattr(df, analysis_fcn)(axis=1),\
                     color=col, label=labels[i+1])
        p.title("Face 2 to 0")
        p.xlim(0, len(xtick_labels)+3)
        locs, xlabels = p.xticks(np.arange(1, len(xtick_labels)+1), xtick_labels, size=8)
        p.setp(xlabels, rotation=70)
        p.ylabel(stat+" Distance")

        if verbose:
            p.show()
        else:
            p.savefig("distance_comparisons_"+stat+".pdf")
            p.close()


def timestep_comparisons(path, verbose=False):
    data_files = [os.path.join(path,x) for x in os.listdir(path) if os.path.isfile(os.path.join(path,x)) \
                  and x[-2:]=="h5"]
    if len(data_files)==0:
        print "The inputed path contains no HDF5 files."
        return None

    data = [HDFStore(filename) for filename in data_files]

    for key in data[0].keys():
        for i, dataset in enumerate(data):
            # p.subplot(3,3,i)
            df = dataset[key].sort(axis=0).sort(axis=1)
            df.T.plot(style="D--")
            p.legend(prop={'size':8}, loc="best")
            p.title(str(key)[1:])
            locs, xlabels = p.xticks(size=8)
            p.setp(xlabels, rotation=70)

            if verbose:
                p.show()
            else:
                p.savefig("timestep_comparisons_"+str(data_files[i][38:-19])+"_"+str(key[1:])+".pdf")
                p.close()


def inverse_factorial(n):
    '''

    Because why not?

    '''
    from scipy.misc import factorial
    x = 1

    while factorial(x) != n:
        x += 1
        if x>n:
            return "n must be a valid whole number."
    return x

def num_fiducials(N):
    '''

    Return the number of fiducials based on the number of lines in the
    comparison file.

    Parameters
    **********

    N : int
        Number of rows in the data frame.

    '''

    n = 1

    while n<N:
        if n*(n-1) == 2*N:
            return n-1
        else:
            n += 1

    return "Doesn't factor into an integer value."
