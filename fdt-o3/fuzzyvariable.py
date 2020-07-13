# ~~~~~~~~~~~~~~~~
# fuzzyvariable.py
# ~~~~~~~~~~~~~~~~

import copy
import numpy as np
from fuzzyset import FuzzySet

class FuzzyVariable(object):

    """Base class for a Fuzzy Variable object.

        A fuzzy variable defines the language that will be used to discuss a
        fuzzy concept such as temperature, pressure, age, or height. The class
        FuzzyVariable is used to create instances of a fuzzy variable, giving
        a name, the universe of discourse for the variable and a set of primary
        fuzzy terms that will be used when describing specific fuzzy concepts
        associated with the fuzzy variable. 

        Parameters
        ----------
        name : string
            Defines the name of the fuzzy variable.

        universe : 1d array
            Defines a set of lower and upper bounds for the values of the fuzzy
            sets used to describe the concepts of the fuzzy variable. 

        terms : array, default=None
            The fuzzy terms are described using a term name such as hot, along
            with a fuzzy set that represents that term.        
    """

    def __init__(self, name='FuzzyVariable', universe=[], terms=[]):
        self.name = name
        self.x = universe
        self.terms = terms
        self._initfuzzy()
        self._initvalues()

    def _initfuzzy(self):
        self.fuzzy = [FuzzySet(self.x) for t in self.terms]

    def _initvalues(self):
        self.values = {t: f for t, f in zip(self.terms, self.fuzzy)}

    def __len__(self):
        return len(self.terms)

    def __call__(self, x):
        """Return the memebership degree of x for every fuzzy set defined."""
        nans = np.isnan(x)
        mval = np.ones((len(x), len(self.fuzzy)))
        for i, v in enumerate(x):
            if nans[i]:
                continue
            else:
                mval[i,:] = [fuzz(v) for fuzz in self.fuzzy]
        return mval.T

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    # Getter functions
    def __getitem__(self, term):
        """Return the fuzzy set corresponding to the term of the FuzzyVariable."""
        return self.values[term]

    def getname(self):
        """Return the name of the FuzzyVariable."""
        return self.name
    
    def getuniverse(self):
        """Return the universe of discourse of the FuzzyVariable."""
        return self.x

    def getterms(self):
        """Return a list with the linguistic values of the FuzzyVariable."""
        return self.terms

    def getfuzzy(self):
        """Return a list with the fuzzy values of the FuzzyVariable."""
        return self.fuzzy

    def getvalues(self, v):
        """Return a dictionary which corresponds linguistic to fuzzy values."""
        return self.values

    def getvagueness(self):
        return [fuzz.vagueness() for fuzz in self.fuzzy]

    # Setter functions
    def setname(self, name):
        """Set the name of the FuzzyVariable."""
        self.name = name

    def setuniverse(self, newX):
        """Set the universe of the FuzzyVariable."""
        self.x = newX
        for fuzz in self.fuzzy:
            fuzz.x = newX

    def setterms(self, terms):
        self.terms = terms
        
    def setfuzzy(self, fuzzy, p=None):
        self.fuzzy = fuzzy
        self.setvalues()
            
    def setvalues(self):
        self.values = {t: f for t, f in zip(self.terms, self.fuzzy)}

    def setmf(self, params):
        """Set membership functions for every FuzzySet."""
        for i, p in enumerate(params):
            self.fuzzy[i].set_mf(p[0], p[-1])

    def setmval(self, mval):
        for j, fuzz in enumerate(self.fuzzy):
            fuzz.set_mval(mval[:, j])

    # Adder functions
    def addterm(self, terms):
        """Add linguistic : fuzzy value pairs to the FuzzyVariable."""
        self.terms.extend(terms)
        for t in terms:
            f = FuzzySet(self.x)
            self.fuzzy.append(f)
            self.values.setdefault(t, f)

    # Other functions
    def ambiguity(self):
        """Return the ambiguity measure of the FuzzyVariable."""
        pval = np.zeros(self.x.shape[0])
        for i, row in enumerate(self._p_distribution()):
            row = np.append(row, 0) / np.max(row)
            diff = -np.diff(row)
            idx = np.arange(1, row.shape[0])
            pval[i] = np.sum(diff * np.log(idx))
        return np.sum(pval) / (pval.shape[0]) / np.log(len(idx))
    
    def _p_distribution(self):
        """Return the possibility distribution of the FuzzyVariable."""
        pdist = np.array([fuzz.m for fuzz in self.fuzzy]).T
        pdist = -np.sort(-pdist, axis=1)
        return pdist

    def acut(self, a):
        for fuzz in self.fuzzy:
            idx = np.where(fuzz.m < a)
            fuzz.m[idx] = 0

### Testing/Debbuging 
##uod = np.arange(0,20)
##m0 = np.arange(1,3,0.1)
##t = ['Low']
##fv = FuzzyVariable('Temperature', np.arange(0, 1, 0.01), t)
##fv.setmval(np.array([m0]).T)
##fv.setmf([('trimf', [0, .5, 1])])



