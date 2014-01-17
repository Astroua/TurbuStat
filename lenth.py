'''

Implementation of Lenth's Method. This returns a t-like statistic allowng the sensitivity
to be examined.

'''

import numpy as np
import matplotlib.pyplot as p
from pandas import Panel, read_hdf, DataFrame, concat, Series
import statsmodels.formula.api as sm


class LenthsMethod(object):
    """This implements Lenth's Method for use in the sensitivity analysis"""
    def __init__(self, datafile, save_name=None, columns=None, rows=None, \
                 statistics=["Wavelet", "MVC", "PSpec", "Genus", "VCS", "VCA", "Tsallis", "Skewness", "Kurtosis"]):
        super(LenthsMethod, self).__init__()

        assert isinstance(datafile, str)
        self.datafile = lambda x: read_hdf(datafile,x)

        self.columns = columns
        self.rows = rows
        self.save_name = save_name
        self.statistics = statistics

        self.paramnames = None

        self.mean_respvecs = []
        self.lnvar_respvecs = []

        self.mean_fitparam = []
        self.lnvar_fitparam = []

        self.mean_signifig = []
        self.lnvar_signifig = []

        self.model_matrix = np.array([[1,-1,1,1,1],
                         [1,1,1,1,1],
                         [1,-1,-1,1,1]])#,
                        #  [1,1,-1,1,1],
                        #  [1,-1,1,-1,1],
                        #  [1,1,1,-1,1],
                        #  [1,-1,-1,-1,1],
                        #  [1,1,-1,-1,1],
                        #  [1,-1,1,1,-1],
                        #  [1,1,1,1,-1],
                        #  [1,-1,-1,1,-1],
                        #  [1,1,-1,1,-1],
                        #  [1,-1,1,-1,-1],
                        #  [1,1,1,-1,-1],
                        #  [1,-1,-1,-1,-1],
                        #  [1,1,-1,-1,-1]
                        # ])

        self.model_matrix = DataFrame(self.model_matrix,columns=["Constant","Mach", "B_field", "Driving", "Temperature"])

    def make_response_vectors(self):

        for i, stat in enumerate(self.statistics):
            data = self.datafile(stat)

            if i==0:
                self.model_matrix.index = data.index # Set the indexes to be the same

            if self.columns is not None:
                data = data[self.columns,:]
            if self.rows is not None:
                data = data[:,self.rows]

            ## Transform to orthogonal parameterization
            data = 2*data - 1

            # Two different ways to define the responses
            # Should yield a 16 element vector for each comparison statistic
            self.mean_respvecs.append(data.mean(axis=1))
            self.lnvar_respvecs.append(np.log(data.std(axis=1)**2.))

        return self

    def fit_model(self, model=None, verbose=False):

        if model is None:
            model = "Mach*B_field*Driving*Temperature"

        for i,(stat, mean, lnvar) in enumerate(zip(self.statistics, self.mean_respvecs, self.lnvar_respvecs)):
            # if i==0:
            #     # model_matrix.append({"mean_resp":mean, "lnvar_resp":lnvar})
            #     self.model_matrix = concat(self.model_matrix[:],{"mean_resp":mean, "lnvar_resp":lnvar})
            # else:
            self.model_matrix["mean_resp"] = Series(mean, index=self.model_matrix.index)
            self.model_matrix["lnvar_resp"] = Series(lnvar, index=self.model_matrix.index)

            mean_model = sm.ols("".join(["mean_resp~",model]), data=self.model_matrix)
            lnvar_model = sm.ols("".join(["lnvar_resp~",model]), data=self.model_matrix)

            mean_results = mean_model.fit()
            lnvar_results = lnvar_model.fit()

            self.mean_fitparam.append(mean_results.params[1:])
            self.lnvar_fitparam.append(lnvar_results.params[1:])

            if i==0:
                self.paramnames = mean_model.exog_names[1:] # Set the names of the coefficients

            if verbose:
                print "Fits for "+ stat
                print mean_results.summary()
                print lnvar_results.summary()
        return self

    def lenth(self, IER=2.16):
        '''
         IER: Wu Hamada critical value 2.16 for 15 compared effect estimate with alpha 0.05
         '''

        for mean, lnvar in zip(self.mean_fitparam, self.lnvar_fitparam):
            s0_mean = np.median(np.abs(mean))
            s0_lnvar = np.median(np.abs(lnvar))

            pse_mean = 1.5*np.median(np.abs(mean)[np.where(np.abs(mean)<=2.5*s0_mean)[0]])
            pse_lnvar = 1.5*np.median(np.abs(lnvar)[np.where(lnvar<=2.5*s0_lnvar)[0]])

            self.mean_signifig.append([mean/pse_mean,pse_mean*IER])
            self.lnvar_signifig.append([lnvar/pse_lnvar, pse_lnvar*IER])

        return self

    def make_plots(self):
        import matplotlib.pyplot as p

        index = np.arange(len(self.paramnames))
        bar_width = 0.35

        for i,stat in enumerate(self.statistics):
            p.subplot(2,1,1)
            p.bar(index, self.mean_signifig[i][0], color="b")
            p.plot([0.,index.max()+2*bar_width],[self.mean_signifig[i][1],self.mean_signifig[i][1]], "b-",
                   [0.,index.max()+2*bar_width],[-self.mean_signifig[i][1],-self.mean_signifig[i][1]], "b-")
            p.xlabel("Effect Estimates")
            p.ylabel(stat)
            p.xticks(index+bar_width, self.paramnames, rotation="vertical", verticalalignment="bottom")
            p.tight_layout()
            p.subplot(2,1,2)
            p.bar(index, self.lnvar_signifig[i][0], color="b")
            p.plot([0.,index.max()+2*bar_width],[self.lnvar_signifig[i][1],self.lnvar_signifig[i][1]], "b-",
                   [0.,index.max()+2*bar_width],[-self.lnvar_signifig[i][1],-self.lnvar_signifig[i][1]], "b-")
            p.xlabel("Effect Estimates")
            p.ylabel(stat)
            p.xticks(index+bar_width, self.paramnames, rotation="vertical", verticalalignment="bottom")
            p.tight_layout()
            p.show()

    def run(self):
        self.make_response_vectors()
        self.fit_model()
        self.lenth()
        self.make_plots()

        return self


