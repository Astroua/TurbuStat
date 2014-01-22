
'''

Implementation of the Cramer Statistic

'''

class Cramer_Distance(object):
    """docstring for Cramer_Distance"""
    def __init__(self, cube1, cube2, data_format, nsamples, alpha):
        super(Cramer_Distance, self).__init__()
        self.cube1 = cube1
        self.cube2 = cube2
        self.data_format = data_format
        self.nsamples = nsamples
        self.alpha = alpha

        self.data_matrix = None
        self.cramer_distance = None

    def format_data(self):
        pass

    def bootstrap(self, nsamples=None, statistic, alpha):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
        if nsamples is not None:
            self.nsamples = nsamples

        shape = self.data_matrix.shape
        idx = []
        for n in shape:
            idx.append(np.random.randint(0, n, (self.nsamples, n)))
        samples = self.data_matrix[idx]
        stat = np.sort(self.statistic(samples))
        self.cramer_distance = (stat[int((alpha/2.0)*self.nsamples)], \
                stat[int((1-alpha/2.0)*self.nsamples)])
        return self

    def run(self, verbose=False):

        if verbose:
            import matplotlib.pyplot as p

        return self

def cramer_statistic(lookup):
    m, n = lookup.shape

    xind, yind = np.indices((m,n))
    m, n = float(m), float(n)

    term1 = (1/(m*n)) * np.sum()

    return (m*n/(m+n)) * (term1 + term2 + term3)
