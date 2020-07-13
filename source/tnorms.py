# ~~~~~~~~~
# tnorms.py
# ~~~~~~~~~

import numpy as np

class tNorms(object):

    def __init__(self, norm):
        self.norm = norm

    def __call__(self, x, y):
        if self.norm is 'minmax':
            return tNorms.tmin(x, y)
        elif self.norm is 'alg':
            return tNorms.algprod(x, y)
        elif self.norm is 'bnd':
            return tNorms.bndprod(x, y)
        elif self.norm is 'dra':
            return tNorms.draprod(x, y)

    @staticmethod 
    def tmin(x, y):
        return np.minimum(x, y)

    @staticmethod
    def algprod(x, y):
        return x * y

    @staticmethod
    def bndprod(x, y):
        return np.maximum(x+y-1, 0)

    @staticmethod
    def draprod(x, y):
        t = np.zeros(len(x))
        idx = np.where(np.logical_or(x==1, y==1))
        t[idx] = np.minimum(x[idx], y[idx])
        return t
        
class sNorms(object):

    def __init__(self, norm):
        self.norm = norm

    def __call__(self, x, y):
        if self.norm is 'minmax':
            return sNorms.tmax(x, y)
        elif self.norm is 'alg':
            return sNorms.algsum(x, y)
        elif self.norm is 'bnd':
            return sNorms.bndsum(x, y)
        elif self.norm is 'dra':
            return sNorms.drasum(x, y)

    @staticmethod 
    def tmax(x, y):
        return np.maximum(x, y)

    @staticmethod     
    def algsum(x, y):
        return x + y - x*y

    @staticmethod 
    def bndsum(x, y):
        return np.minimum(1, x+y)

    @staticmethod     
    def drasum(x, y):
        t = np.ones(len(x))
        idx = np.where(np.logical_or(x==0, y==0))
        t[idx] = np.maximum(x[idx], y[idx])
        return t


### Testing/Debugging
##np.random.seed(0)
##t = tNorms('dra')
##s = sNorms('dra')
##x = np.random.random(5)
##y = np.random.random(5)
##print(x, y, sep='\n')
##print(t(x, y))
##print(s(x, y))
