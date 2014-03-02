
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
import os


class DendroStats(object):
    """

    docstring for DendroStats

    Parameters
    **********

    save : bool, optional
          Sets whether each dendrogram should be saved (in HDF5). If True,
          dendrograms are loaded in only when needed. If False, the entire
          dendrogram is kept open.

    """
    def __init__(self, cube, header, deltas=None, save=True, save_name=None, \
                 save_path=None, verbose=True):
        super(DendroStats, self).__init__()
        self.cube = cube
        self.header = header
        self.save = save

        if deltas is None:
            pass
            ## define some criterion to set uniform delta values to test
        else:
            assert isinstance(deltas, np.ndarray)
            self.deltas = deltas

        if self.save:
            if save_name is None:
                print "Specify save name for dendrogram output. Set to \
                        generic name with header's object name if available."
                try:
                    save_name = header["OBJECT"]
                except KeyError:
                    print "No object name found in the header."
                    save_name = "generic_savename"


        for delta in self.deltas:
            dendro_file = load_dendro(delta, save_name)
            if save:
                if dendro_file is None:
                    self.dendrograms[delta] = compute_dendro(cube, header, delta,\
                                     save=save, save_name=save_name, verbose=verbose)
                else:
                    self.dendrograms[delta] = dendro_file
            else:
                if dendro_file is None:
                    self.dendrograms[delta] = compute_dendro(cube, header, delta,\
                                     save=save, save_name=save_name, verbose=verbose)
                else:
                    self.dendrograms[delta] = Dendrogram.load_from(dendro_file)


        self.num_features = np.empty((1, max(self.deltas.shape)))
        self.histogram = {}



    def num_features(self):
        '''

        Compute the number of leaves and branches in a dendrogram at all
        specified values of delta.

        '''

        for delta, i in enumerate(self.deltas):
            if self.save:
               d  = Dendrogram.load_from(self.dendrograms[delta])
            else:
                d = self.dendrograms[delta]

            self.num_features[:,i] = d.__len__()

        return self

    def make_histogram(self):
        '''

        Returns a histogram of the intensities of the leaves and branches.

        '''

        for delta, i in enumerate(self.deltas):
            if self.save:
               d  = Dendrogram.load_from(self.dendrograms[delta])
            else:
                d = self.dendrograms[delta]

        return self



def compute_dendro(cube, header, delta, save_name, verbose=False, save=True,\
                   save_path=None): ## Pass **kwargs into Dendrogram class??
    '''

    Computes a dendrogram and (optional) save it in HDF5 format.

    '''
    dendrogram = Dendrogram.compute(cube, delta=delta, verbose=verbose)

    if save:
        dendrogram.save_to(save_name+"_"+str(delta)+"_dendrogram.hdf5")

        return os.join(save_path, save_name+"_"+str(delta)+"_dendrogram.hdf5")

    return dendrogram

def load_dendro(delta, save_name, save_path=None):
    '''

    Returns the file name if the file exists.

    '''

    if save_path is None:
        save_path = ""

    if save_name+"_"+str(delta)+"_dendrogram.hdf5" in os.listdir(save_path):
        dendrogram_files = os.join(save_path, save_name+"_"+str(delta)+"_dendrogram.hdf5")3
        return dendrogram_file
    else:
        return None

def is_individual():
        pass

