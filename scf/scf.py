
'''

Implements the Spectral Correlation Function and its Distance
Metric (Rosolowsky et al, 1999).

'''

import numpy as np

class SCF(object):
    """docstring for SCF"""
    def __init__(self, cube, size=11):
        super(SCF, self).__init__()
        self.cube = cube
        if size%2==0:
            print "Size must be odd. Reducing size to next lowest odd number."
            self.size = size - 1
        else:
            self.size = size

        self.scf_surface = np.zeros((self.size, self.size))
        self.dist = None

    def compute_scf(self):
        '''

        '''
        dx = np.arange(self.size)-self.size/2
        dy = np.arange(self.size)-self.size/2

        a,b = np.meshgrid(dx,dy)
        self.dist = np.sqrt(a**2+b**2)

        for i in dx:
            for j in dy:
                tmp = np.roll(self.cube,i,axis=1)
                tmp = np.roll(tmp,j,axis=2)
                values = np.nansum(((self.cube-tmp)**2),axis=0) / \
                                    (np.nansum(self.cube**2,axis=0) + np.nansum(tmp**2,axis=0))

                scf_value = 1. - np.sqrt(np.nansum(values) / np.sum(np.isfinite(values)))
                self.scf_surface[i+self.size/2,j+self.size/2] = scf_value

        return self

    def run(self, verbose=False):

        self.compute_scf()

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(2,1,1)
            p.imshow(self.scf_surface, origin="lower", interpolation="nearest")
            p.colorbar()

            p.subplot(2,1,2)
            p.hist(self.scf_surface.ravel())

            p.show()


class SCF_Distance(object):
    """docstring for SCF_Distance"""
    def __init__(self, cube1, cube2, size=11, fiducial=None):
        super(SCF_Distance, self).__init__()
        self.cube1 = cube1
        self.cube2 = cube2
        self.size = size

        if fiducial is not None:
            self.scf1 = fiducial
        else:
            self.scf1 = SCF(self.cube1, self.size)
            self.scf1.run()

        self.scf2 = SCF(self.cube2, self.size)
        self.scf2.run()

        self.distance = None


    def distance_metric(self, verbose=False):

        difference = (self.scf1.scf_surface - self.scf2.scf_surface)**2.
        self.distance = np.sqrt(np.nansum(difference)/np.sum(np.isfinite(difference)))

        if verbose:
            import matplotlib.pyplot as p

            # print "Distance: %s" % (self.distance)

            p.subplot(1,3,1)
            p.imshow(self.scf1.scf_surface, origin="lower", interpolation="nearest")
            p.title("SCF1")
            p.colorbar()
            p.subplot(1,3,2)
            p.imshow(self.scf2.scf_surface, origin="lower", interpolation="nearest")
            p.title("SCF2")
            p.colorbar()
            p.subplot(1,3,3)
            p.imshow(difference, origin="lower", interpolation="nearest")
            p.title("Difference")
            p.colorbar()

            p.show()

        return self
