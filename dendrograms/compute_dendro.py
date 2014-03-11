
'''

Dendrogram statistics as described in Burkhart et al. (2013)
Two statistics are contained:
    * number of leaves + branches vs. $\delta$ parameter
    * statistical moments of the intensity histogram

Requires the astrodendro package (http://github.com/astrodendro/dendro-core)

'''

from astrodendro import Dendrogram
import numpy as np
from matplotlib import pyplot as p
import matplotlib.cm as cm
from pandas import Series, DataFrame

## Load in utilities. Change to import once installed as package
execfile("../utilities.py")

def DendroStats(path, min_deltas):
    '''
    '''

    numfeatures = []
    histograms = []
    min_npix=10
    min_value=0.001
    verbose=False
    keywords = {"centroid", "centroid_error", "integrated_intensity", "integrated_intensity_error", "linewidth",\
                 "linewidth_error", "moment0", "moment0_error", "cube"}

    cube, header = fromfits(path, keywords)["cube"]

    for i, delta in enumerate(min_deltas):
        d = Dendrogram.compute(cube, min_delta=delta, min_npix=min_npix,\
                               min_value=min_value, verbose=verbose)

        numfeatures.append(d.__len__())
        histograms.append([branch.vmax for branch in d.branches]+[leaf.vmax for leaf in d.leaves])
        # print "On %s/%s" % (i+1, len(min_deltas))

    min_deltas = Series(min_deltas)
    numfeatures = Series(numfeatures)
    histograms = Series(histograms)

    df = DataFrame({"Deltas": min_deltas, "Num Features": numfeatures, "Histograms": histograms})

    return df

def single_input(a):
    return DendroStats(*a)

if __name__ == "__main__":

    import os
    import sys
    from multiprocessing import Pool
    from itertools import izip, repeat
    from pandas import HDFStore

    path = sys.argv[1]
    ncores = int(sys.argv[2])

    min_deltas = np.logspace(-1.5,-0.3,20)

    timesteps = [os.path.join(path,x) for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]

    pool = Pool(processes=ncores)
    dataframes = pool.map(single_input, izip(timesteps, repeat(min_deltas)))
    pool.close()
    pool.join()

    timesteps_labels = [x[-8:] for x in timesteps]

    run_label = path.split("/")[-2]

    store = HDFStore(path+run_label+"_dendrostats.h5")

    for df, label in izip(dataframes, timesteps_labels):
        store[label] = df

    store.close()