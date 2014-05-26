'''

The density PDF as described by Kowal et al. (2007)

'''

import numpy as np
from scipy.stats import nanmean

def pdf(img, num_bins=1000, verbose=True):
    '''

    Creates the PDF given an image (of any dimension)

    INPUTS
    ------

    img - array
          n-dim array

    OUTPUTS
    -------

    '''

    img_av = nanmean(img, axis=None)  ## normalize by average
    hist, edges = np.histogram(img/img_av,bins=num_bins,density=True)
    hist /= np.sum(~np.isnan(img))
    bin_centres = (edges[:-1] + edges[1:])/2

    if verbose:
        import matplotlib.pyplot as p
        p.grid(True)
        p.loglog(bin_centres, hist, 'bD-')
        p.xlabel(r"$\Sigma/\overline{\Sigma}$")
        p.ylabel("PDF")
        p.show()

    return bin_centres, hist
