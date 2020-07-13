# ~~~~~~~~~
# automf.py
# ~~~~~~~~~

import numpy as np

class AutoMF(object):

    """Automatic membership function generator.

        This function calculates the parameters for a certain family of functions,
        in order to assing membership functions to FuzzyVariable objects.

        Parameters
        ----------
        X : 1d array, size(N)
            Initial array of data where N is the number of samples.

        func_type : {'trimf', 'trapmf', 'gaussmf'}, default='trimf'
            pass
            
        c : list, default=[2, 7]
            Contains the number of centers

        cV : list, default=[2, 7]
            Contains the number of centers

        U : list, default=[2, 7]
            Contains the number of centers

    """

    term_set = {2: ['Low', 'High'],
                3: ['Low', 'Average', 'High'],
                4: ['Low', 'Average', 'High', 'Very High'],
                5: ['Low', 'Below Average', 'Average', 'High', 'Very High'],
                6: ['Low', 'Below Average', 'Average', 'Above Average', 'High', 'Very High'],
                7: ['Very Low', 'Low', 'Below Average', 'Average', 'Above Average', 'High', 'Very High'],
                8: ['Very Low', 'Low', 'Below Average', 'Average', 'Above Average', 'High', 'Very High', 'Extremly High'],
                9: ['Extremly Low', 'Very Low', 'Low', 'Below Average', 'Average', 'Above Average', 'High', 'Very High', 'Extremly High']}
                #8: ['N','NE','E','SE','S','SW','W','NW']}
            
    def __init__(self, X, func_type='trimf', c=None, cV=None, U=None):
        self.X = X.reshape((len(X), 1))
        self.N = len(X)
        self.func_type = func_type
        self.c = c
        self.cV = cV
        self.U = U
        self._set_mf()    

    def _set_mf(self):
        if self.func_type == 'singlemf':
            prmt = self.cV
            self.pV = [(self.func_type, p) for p in prmt]
        if self.func_type == 'trimf':
            prmt = np.array(self.trimf_prmts()).reshape((3, self.c)).T
            self.pV = [(self.func_type, p) for p in prmt]
        if self.func_type == 'trapmf':
            prmt = np.array(self.trapmf_prmts()).reshape((4, self.c)).T
            self.pV = [(self.func_type, p) for p in prmt]

    def trimf_prmts(self):
        sX_ind = np.argsort(self.X[:, 0])
        sX = self.X[sX_ind]
        cV = self.cV
        sU = self.U[sX_ind, :]
        aV = []
        bV = []
        gV = []
        for i in range(self.c):
            minU = np.min(sU[:, i]) + 5e-2
            lcut = np.where(sU[:, i] <= minU)[0]
            maxU = np.max(sU[:, i]) - 5e-2
            ucut = np.where(sU[:, i] >= maxU)[0]
            # get beta parameter
            L, R = sX[ucut[0]], sX[ucut[-1]]
            beta = (L+R)/2
            bV.append(beta)
            # get alpha parameter
            if i == 0:
                alpha = beta
            else:
                acut = np.where(sU[:ucut[0], i] <= minU)[0]
                if acut.size == 0:
                    x0 = sX[0]
                    alpha = x0 - (sU[0, i]/(1-sU[0, i]))*(beta-x0)
                else:
                    alpha = sX[acut[-1]]
            aV.append(alpha)
            # get gamma parameter
            gcut = np.where(sU[ucut[-1]:, i] <= minU)[0] + ucut[-1]
            if i == self.c-1:
                gamma = beta
            else:
                if gcut.size ==0:
                    x0 = sX[-1]
                    gamma = x0 + (sU[-1, i]/(1-sU[-1, i]))*(x0-beta)
                else:
                    gamma = sX[gcut[0]]
            gV.append(gamma)
        return aV, bV, gV

    def trapmf_prmts(self):
        sX_ind = np.argsort(self.X[:, 0])
        sX = self.X[sX_ind]
        cV = self.cV
        sU = self.U[sX_ind, :]
        aV = []
        bV = []
        gV = []
        dV = []
        for i in range(self.c):
            minU = np.min(sU[:, i]) + 5e-2
            lcut = np.where(sU[:, i] <= minU)[0]
            maxU = np.max(sU[:, i]) - 5e-2
            ucut = np.where(sU[:, i] >= maxU)[0]
            # get beta and gamma parameters
            L, R = sX[ucut[0]], sX[ucut[-1]]
            beta = L
            gamma = R
            # get alpha parameter
            if i == 0:
                beta = gamma
                alpha = beta
            else:
                acut = np.where(sU[:ucut[0], i] <= minU)[0]
                if acut.size == 0:
                    x0 = sX[0]
                    alpha = x0 - (sU[0, i]/(1-sU[0, i]))*(beta-x0)
                else:
                    alpha = sX[acut[-1]]
            # get delta parameter
            dcut = np.where(sU[ucut[-1]:, i] <= minU)[0] + ucut[-1]
            if i == self.c-1:
                beta = gamma
                delta = gamma
            else:
                if dcut.size ==0:
                    x0 = sX[-1]
                    delta = x0 + (sU[-1, i]/(1-sU[-1, i]))*(x0-gamma)
                else:
                    delta = sX[dcut[0]]
            aV.append(alpha)
            bV.append(beta)
            gV.append(gamma)
            dV.append(delta)
        return aV, bV, gV, dV


##from sklearn.datasets import load_wine
### Initialize the dataset
##wine = load_wine()
##X = wine.data[:, 7]
##var_names = wine.feature_names
##a = AutoMF(X)
