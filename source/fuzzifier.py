# ~~~~~~~~~~~~
# fuzzifier.py
# ~~~~~~~~~~~~

import csv
import copy
import pickle
import numpy as np
import pandas as pd
from cmeans import cMeans
from automf import AutoMF
from fuzzyvariable import FuzzyVariable
          
    
class Fuzzifier(object):

    """Dataset fuzzification algorithm.

        Fuzzification is the process of converting a crisp input value to
        a fuzzy value, that is performed by the use of the information in
        the knowledge base. Although various types of curves can be seen
        in literature, gaussian, triangular, and trapezoidal MFs are the
        most commonly used in the fuzzification process.

        Parameters
        ----------
        data : 2d array, size(S, N)
            Data to be fuzzified. S is the number of samples and N is the
            number of features within each sample vector.

        target : 1d array, size(S)
            Contains the values of the target set. The fuzzification process
            of this dataset will be determined by the task parameter.

        task : {'clf', 'reg'}, default='clf'
            Defines the purpose the dataset will be used for. It is relevant
            due to the fact that classification targets must be fuzzified to
            singleton membeship functions

        Attributes
        ----------
        best_c : dict
            Contains the optimum number of clusters for every feature.
            
        best_cntrs : dict
            Contains the coordinates of the cluster centers in acsending order
            for every feature.

        best_Umat : dict
            Contains the partition matrix for every feature. Every column of
            this matrix contains the membership degree of every element with
            respect to the corresponding cluster center.

        fuzzy_var : dict
            Contains the FuzzyVariable objects that correspond to each feature.

        fuzzy_data : dict
            Contains a FuzzyDataset object.
    """

    def __init__(self, data, target, task='clf'):

        # Check for defective input values (mainly for debugging)
        assert data.ndim == 2, \
               'input data must be a 2d array of size SxN.'

        assert target.ndim == 1, \
               'target data must be a 1d array of size S.'

        assert task in ['clf', 'reg'], \
               'task parameter is not properly defined.'
        
        self.X = data
        self.S, self.N = data.shape
        self.y = target
        self.task = task
        
    def get_clusters(self, best_c=[3, 4], init_c=[], m=2, error=5e-3, maxiter=100,
                     metric='euclidean', fpcoef='PC'):
        """Call the Fuzzy c-Means algorithm to find the best clusters for 
           every feature of the dataset plus the target set.

            For more information check the cMeans documentation.

            Parameters
            ----------
            best_cntrs : list, default=[3, 4]
                Contains the minimum and maximum number of clusters to be
                searched.

            init_cntrs : list, deafult=[]
                Contains the exact number of clusters for every fuzzy variable.
                
            m : float
                The weighting exponent m > 1.

            error : float
                Termination criterion.

            maxiter : int
                Maximum number of iterations allowed.

            metric: {'euclidean', 'diagonal', 'mahalanobis'}
                The norm inducing matrix A.

            fpcoef : {'PC', 'PE', 'XB'}
                The fuzzy partition coefficient.
        """
        self.best_c = {}
        self.best_cntrs = {}
        self.best_Umat = {}
        
        # Run for every feature in the dataset
        for i, X in enumerate(self.X.T):
            x = X[~np.isnan(X)]
            print('##Running FCM for attribute {}...'.format(i))
            # Calculate the optimum number of clusters
            temp = []
            if init_c != []:
                cntrs = [init_c[i]]
            else:
                cntrs = list(range(best_c[0], best_c[-1]+1))
            for c in cntrs:
                info = cMeans(x.reshape((len(x), 1)), c, m, error, maxiter, metric, fpcoef)
                temp.append(info.fpc)
            self.best_c[i] = cntrs[temp.index(max(temp))]
            # Call FCM for best_c clusters and store the necessary information
            best = cMeans(x.reshape((len(x), 1)), self.best_c[i], m, error, maxiter)
            ind = np.argsort(best.centers, axis=0).flatten()            
            self.best_cntrs[i] = best.centers[ind]
            self.best_Umat[i] = best.U[:, ind]

        # Run for the target set
        print('##Running FCM for target...')
        if self.task == 'clf':
            self.best_c[-1] = len(set(self.y))
        elif self.task == 'reg':
            temp = []
            if init_c != []:
                cntrs = [init_c[-1]]
            else:
                cntrs = list(range(best_c[0], best_c[-1]+1))
            y = self.y[~np.isnan(self.y)]
            for c in cntrs:
                info = cMeans(y.reshape((len(y), 1)), c, m, error, maxiter, metric, fpcoef)
                temp.append(info.fpc)
            self.best_c[-1] = cntrs[temp.index(max(temp))]
        best = cMeans(y.reshape((len(y), 1)), self.best_c[-1], m, error, maxiter)
        ind = np.argsort(best.centers, axis=0).flatten()            
        self.best_cntrs[-1] = best.centers[ind]
        self.best_Umat[-1] = best.U[:, ind]

    def get_fuzzy_variables(self, func_type='trimf', feat_names=None):
        """Call the AutoMF generator to assign membership functions to every
           fuzzy variable. For more information check the AutoMF documentation.

            Parameters
            ----------
            func_type : {'trimf', 'trapmf', 'gaussmf', 'noapp'}, default='trimf'
                Defines the parametric form of the membership functions.

            feat_names : array, default=None
                Contains the names of the FuzzyVariable objects. If None, the
                names will have the form: 'Var_i'.
        """
        self.fuzzy_var = {}
        # Get feature names or create them
        if feat_names is None:
            feat_names = ['Var_{}'.format(i) for i in range(self.N)]
            feat_names.append('Var_t')
        else:
            feat_names = list(feat_names)
            feat_names.append('Target Variable')
        # Run for every feature in the dataset
        for i, X in enumerate(self.X.T):
            x = X[~np.isnan(X)]
            # Call AutoMF to create FuzzyVariables and assign MFs
            a = AutoMF(x, func_type, self.best_c[i], self.best_cntrs[i], self.best_Umat[i])
            umin, umax = 0.8*min(x), 1.2*max(x)
            step = (umax-umin)/1000
            universe = np.arange(umin, umax, step)
            terms = a.term_set[self.best_c[i]]          
            fvar = FuzzyVariable(feat_names[i], universe, terms)
            fvar.setmf(a.pV)
            mval = fvar(X)
            fvar.setmval(mval.T)
            self.fuzzy_var[i] = fvar
            
        # Create a FuzzyVariable for the target set
        y = self.y[~np.isnan(self.y)]
        if self.task == 'clf':
            a = AutoMF(y.reshape((len(y), 1)), 'singlemf', self.best_c[-1], self.best_cntrs[-1], self.best_Umat[-1])
            universe = np.array(list(set(y)))
        elif self.task == 'reg':
            a = AutoMF(y.reshape((len(y), 1)), func_type, self.best_c[-1], self.best_cntrs[-1], self.best_Umat[-1])
            universe = np.arange(int(0.8*min(y)), int(1.2*max(y)), 2e-1)
        terms = ['Class {}'.format(i) for i in range(self.best_c[-1])]          
        fvar = FuzzyVariable(feat_names[-1], universe, terms)
        fvar.setmf(a.pV)
        mval = fvar(self.y)
        fvar.setmval(mval.T)
        self.fuzzy_var[-1] = fvar

    def set_mf(self, key, mf):
        mfunc = mf[0]
        prmts = mf[-1]
        self.fuzzy_var[key].setmf(mf)
            
    def get_fuzzy_data(self):
        """Create a FuzzyDataset object."""
        fdata = {}
        for i, fvar in enumerate(self.fuzzy_var.values()):
            if i == self.N:
                fdata[-1] = np.array(fvar(self.y)).T
            else:
                fdata[i] = np.array(fvar(self.X[:, i])).T
        self.fuzzy_dataset = FuzzyDataset(fdata, self.fuzzy_var)


class FuzzyDataset(object):

    """Container of a fuzzified dataset.

        This class contains helper functions, in order to properly manipulate
        a fuzzy dataset.
        
        Parameters
        ----------
        fdata : dict
            pass

        fvar : array
            pass
    """
    
    def __init__(self, fdata={}, fvar={}):
        self.fuzzy_dataset = copy.deepcopy(fdata)
        self.fuzzy_variables = copy.deepcopy(fvar)

    def __repr__(self):
        return 'FuzzyDataset'

    def __str__(self):
        return 'FuzzyDataset'

    def __len__(self):
        return len(self.fuzzy_variables)

    def store_db(self, fname):
        with open(str(fname)+'.csv', 'w', newline="") as f:
            w = csv.writer(f, delimiter=',')
            for v in self.fuzzy_dataset.values():
                w.writerow(v)
        self._store_fvar(str(fname)+'-fvar.pickle')

    def _store_fvar(self, fname):
        with open(fname, 'wb') as fout:
            pickle.dump(self.fuzzy_variables, fout, pickle.HIGHEST_PROTOCOL)

    def load_db(self, fname):
        with open(str(fname)+'.csv', 'r') as fin:  
            reader = csv.reader(fin)   
            for i, row in enumerate(reader):
                self.fuzzy_dataset[i] = []
                for e in row:
                    e = np.fromstring(e[1:-1], dtype=np.float, sep=' ')
                    self.fuzzy_dataset[i].append(e)
                self.fuzzy_dataset[i] = np.array(self.fuzzy_dataset[i])
            self.fuzzy_dataset[-1]  = self.fuzzy_dataset[i] 
            del self.fuzzy_dataset[i]
            self._get_fuzzyvar(str(fname)+'-fvar.pickle')
            
    def _get_fuzzyvar(self, fname):
        with open(fname, 'rb') as fin:
            self.fuzzy_variables = pickle.load(fin)
                    
            
### Testing/Debugging
##from sklearn.datasets import load_iris, load_wine
### Initialize the dataset
##iris = load_iris()
##X = iris.data
##y = iris.target
##var_names = iris.feature_names
### Fuzzify the dataset
##fuzz = Fuzzifier(X, y)
##fuzz.get_clusters(best_c=[3,4])
##fuzz.get_fuzzy_variables(feat_names=var_names)
##fuzz.get_fuzzy_data()
##fdata = fuzz.fuzzy_dataset
####fdata.store_db('test-iris')
####fdb = FuzzyDataset()
####fdb.load_db('test-iris')
