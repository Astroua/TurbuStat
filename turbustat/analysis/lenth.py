# Licensed under an MIT open source license - see LICENSE


'''

Implementation of Lenth's Method. This returns a t-like statistic allowng the sensitivity
to be examined.

'''

import numpy as np
import matplotlib.pyplot as p
from pandas import read_hdf, DataFrame, Series
import statsmodels.formula.api as sm
import types

class LenthsMethod(object):
    """This implements Lenth's Method for use in the sensitivity analysis"""
    def __init__(self, datafile, save_name=None, columns=None, rows=None, analysis_fcn="mean", \
                 statistics=["Wavelet", "MVC", "PSpec","Bispectrum","DeltaVariance","Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF",
                  "Cramer", "Skewness", "Kurtosis"]):
        super(LenthsMethod, self).__init__()

        assert isinstance(datafile, str)
        self.datafile = lambda x: read_hdf(datafile,x)

        self.columns = columns
        self.rows = rows
        self.save_name = save_name
        self.statistics = statistics

        if isinstance(analysis_fcn, str):
            assert isinstance(getattr(self.datafile(self.statistics[0]), analysis_fcn), types.UnboundMethodType)
        self.analysis_fcn = analysis_fcn



        self.paramnames = None

        self.respvecs = []
        self.laststep_respvecs = []

        self.fitparam = []
        self.laststep_fitparam = []

        self.signifig = []
        self.laststep_signifig = []

        self.model_matrix = np.array([#[1, 1, 1, 1, 1],
                                      [1,-1,-1, 1, 1],
                                      [1, 1,-1, 1,-1],
                                      #[1,-1, 1, 1,-1],
                                      [1, 1,-1,-1, 1],
                                      [1,-1, 1,-1, 1],
                                      [1, 1, 1,-1,-1],
                                      [1,-1,-1,-1,-1]])


        self.model_matrix = DataFrame(self.model_matrix,columns=["Constant","M", "B", "k", "T"])

    def make_response_vectors(self):

        for i, stat in enumerate(self.statistics):
            data = self.datafile(stat).sort(axis=0).sort(axis=1)

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
            if isinstance(self.analysis_fcn, str):
                self.respvecs.append(getattr(data,self.analysis_fcn)(axis=1))
            else:
                self.respvecs.append(self.analysis_fcn(data))
            self.laststep_respvecs.append(data.iloc[:,-1])

        return self

    def fit_model(self, model=None, verbose=False):

        if model is None:
            # model = "Mach*B_field*Driving*Temperature" ## Full
            model = "M+B+k+T+M:B+M:T+B:T" #Fractional

        for i,(stat, vec, last) in enumerate(zip(self.statistics, \
                        self.respvecs, self.laststep_respvecs)):

            self.model_matrix["resp"] = Series(vec, index=self.model_matrix.index)
            self.model_matrix["laststep_resp"] = Series(last, index=self.model_matrix.index)

            fcn_model = sm.ols("".join(["resp~",model]), data=self.model_matrix)
            laststep_model = sm.ols("".join(["laststep_resp~",model]), data=self.model_matrix)

            results = fcn_model.fit()
            laststep_results = laststep_model.fit()

            self.fitparam.append(results.params[1:])
            self.laststep_fitparam.append(laststep_results.params[1:])

            if i==0:
                self.paramnames = fcn_model.exog_names[1:] # Set the names of the coefficients

            if verbose:
                print "Fits for "+ stat
                print results.summary()
                print laststep_results.summary()
        return self

    def lenth(self, IER=2.3):
        '''
         IER: Wu Hamada critical value 2.16 for 15 compared effect estimate with alpha 0.05
         IER: Wu Hamada critical value 2.3 for 7 compared effect estimate with alpha 0.05
         '''

        for param, last in zip(self.fitparam, self.laststep_fitparam):
            s0 = np.median(np.abs(param))
            s0_laststep = np.median(np.abs(last))

            pse = 1.5*np.median(np.abs(param)[np.where(np.abs(param)<=2.5*s0)[0]])
            pse_laststep = 1.5*np.median(np.abs(last)[np.where(last<=2.5*s0_laststep)[0]])

            self.signifig.append([param,pse*IER])
            self.laststep_signifig.append([last, pse_laststep*IER])

        return self

    def make_plots(self):
        import matplotlib.pyplot as p

        bar_width = 0.35
        index = np.arange(len(self.paramnames))

        for i,stat in enumerate(self.statistics):

            p.bar(index, self.signifig[i][0], bar_width, color="b", label="Average")
            p.bar(index+bar_width, self.laststep_signifig[i][0], bar_width, color="g", label="Last")
            p.legend()
            p.plot([0.,index.max()+2*bar_width],[self.signifig[i][1],self.signifig[i][1]], "b-",
                   [0.,index.max()+2*bar_width],[-self.signifig[i][1],-self.signifig[i][1]], "b-")
            p.xlabel("Effect Estimates")
            p.ylabel(stat)
            p.xticks(index+bar_width, self.paramnames, rotation="vertical",
                verticalalignment="top")
            p.tight_layout()
            p.show()

    def run(self, model=None, verbose=False):
        self.make_response_vectors()
        self.fit_model(model=model, verbose=verbose)
        self.lenth()
        self.make_plots()

        return self


