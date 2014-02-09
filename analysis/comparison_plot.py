
import numpy as np
from pandas import read_hdf
import types, os
import matplotlib.pyplot as p
import matplotlib.cm as cm

def comparison_plot(path, analysis_fcn="mean", statistics=["Wavelet", "MVC", \
                    "PSpec","Bispectrum","DeltaVariance","Genus", "VCS", "VCA", \
                    "Tsallis", "PCA", "SCF", "Cramer", "Skewness", "Kurtosis"]):
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

    data_files = [os.path.join(path,x) for x in os.listdir(path) if os.path.isfile(os.path.join(path,x)) and x[-2:]=="h5"]
    if len(data_files)==0:
        print "The inputed path contains no HDF5 files."
        return None

    loader = lambda data, x: read_hdf(data,x)

    if isinstance(analysis_fcn, str):
        assert isinstance(getattr(loader(data_files[0], statistics[0]), analysis_fcn), types.UnboundMethodType)

    for i, stat in enumerate(statistics):
        data_face0 = []
        fid_data_face0 = []

        data_face2 = []
        fid_data_face2 = []

        num_sims = 0
        num_fids = 0

        for data in data_files:
            datum = loader(data, stat).sort(axis=1)
            if data[-21]=="0":
                if data.split("/")[-1][:8]=="fiducial":
                    num_fids = inverse_factorial(datum.shape[0])
                    assert isinstance(num_fids, int)
                    j_last = 0
                    for j in np.arange(num_fids,0, -1):
                        fid_data_face0.append(datum.iloc[j_last:j+j_last,:])
                        j_last = j

                else:
                    data_face0.append(datum.sort(axis=0))
                    num_sims += 1
            elif data[-21]=="2":
                if data.split("/")[-1][:8]=="fiducial":
                    num_fids = inverse_factorial(datum.shape[0])
                    assert isinstance(num_fids, int)
                    j_last = 0
                    for j in np.arange(num_fids,0, -1):
                        fid_data_face2.append(datum.iloc[j_last:j+j_last,:])
                        j_last = j
                else:
                    data_face2.append(datum.sort(axis=0))
                    num_sims += 1
            else:
                print "Check filename for position of face label. Currently works only for \
                            the default output of output.py"
                break

        ## Comparison across simulations

        labels = ["Fiducial "+str(num) for num in range(1,num_fids+2)]
        xtick_labels = ["Design "+str(num) for num in range(1,num_sims+1)]
        xtick_labels = xtick_labels + labels[:-1]

        colours = cm.rainbow(np.linspace(0, 1, len(data_face0)+3))
        ax1 = p.subplot(1,2,1)
        for i, (df, col) in enumerate(zip(data_face0,colours[:num_sims+1])):
            p.scatter(np.arange(1, df.shape[0]+1), getattr(df, analysis_fcn)(axis=1), color=col, label=labels[i])
        p.legend(prop={'size':8}) # make now to avoid repeats
        for i, (df, col)in enumerate(zip(fid_data_face0, colours[:3])):
            p.scatter(np.arange(num_sims+1, num_sims+(num_fids+1-i)), getattr(df, analysis_fcn)(axis=1),\
                     color=col, label=labels[i+1])
        p.title("Face 0")
        p.xlim(0, len(xtick_labels)+3)
        locs, xlabels = p.xticks(np.arange(1, len(xtick_labels)+1), xtick_labels, rotation="vertical", size=8)
        p.setp(xlabels, rotation=70)
        p.ylabel(stat+" Distance")

        ax2 = p.subplot(1,2,2)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        for i, (df, col) in enumerate(zip(data_face2,colours[:num_sims+1])):
            p.scatter(np.arange(1, df.shape[0]+1), getattr(df, analysis_fcn)(axis=1), color=col, label=labels[i])
        p.legend(prop={'size':8}) # make now to avoid repeats
        for i, (df, col)in enumerate(zip(fid_data_face2, colours[:3])):
            p.scatter(np.arange(num_sims+1, num_sims+(num_fids+1-i)), getattr(df, analysis_fcn)(axis=1),\
                     color=col, label=labels[i+1])
        p.title("Face 2")
        p.xlim(0, len(xtick_labels)+3)
        locs, xlabels = p.xticks(np.arange(1, len(xtick_labels)+1), xtick_labels, size=8)
        p.setp(xlabels, rotation=70)
        p.ylabel(stat+" Distance")
        # p.show()
        p.savefig("distance_comparisons_"+stat+".pdf")
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
