# ~~~~~~~~~~~
# fuzzyset.py
# ~~~~~~~~~~~

import copy
import numpy as np
from tnorms import tNorms, sNorms
from membership import MembershipFunc

class FuzzySet(object):

    """Base class for a Fuzzy Set object.

        Fuzzy set is a mathematical model of vague qualitative or quantitative
        data, frequently generated by means of the natural language. The model
        is based on the generalization of the classical concepts of set and 
        its characteristic function.

        Parameters
        ----------
        universe : 1d array
            The universe of discourse on which the Fuzzy Set is defined.

        membership_list : 1d array, default=None
            The membership degree of every element of the universe of
            discourse. If None a membership function must be defined.

        norm : {'minmax', 'alg', 'bnd', 'dra'}, default='minmax'
            pass
    """

    def __init__(self, universe, membership_degrees=None, norm='minmax'):

        # Initialize the universe of discourse
        try:
            self.x = np.asarray(universe)
            self.num_x = np.asarray(universe)
            self.n = len(self.x)
        except:
            raise ValueError('the universe is not properly defined.')

        # Initialize membership degrees for the elements of the fuzzy set
        self._has_mval = False
        self._has_mfunc = False
        if membership_degrees is None:
            self.m = None
        else:
            assert len(membership_degrees) == self.n, \
                   'a membership degree must be defined for every x in UoD.'
            try:
                self.m = np.array(membership_degrees)
                self._has_mval = True
            except:
                raise ValueError('membership degrees list is not properly defined.')
        

        # Initialize norm operators
        assert norm in {'minmax', 'alg', 'bnd', 'dra'}, \
               'the norm operator is not well defined.'
        self.tnorm = tNorms(norm)
        self.snorm = sNorms(norm)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
            
    def set_mf(self, mfunc, prmts):
        """Set a membership function for the FuzzySet object. For more information
           check the MembershipFunc documentation.

            Parameters
            ----------
            mfunc : {str}, default=None
                Defines the type of the membership function. A short description
                for the supported types is given below.
            
              - 'gaussmf'  : Gaussian membership function    - no_prmt=2
              - 'gbellmf'  : Bell-shaped membership function - no_prmt=3
              - 'sigmf'    : Sigmoid membership function     - no_prmt=2
              - 'singlemf' : Singleton membership function   - no_prmt=1
              - 'trapmf'   : Trapezoidal membership function - no_prmt=4
              - 'trimf'    : Triangular membership function  - no_prmt=3

            prmt : list
                Defines the required parameters for the mfunc type membership
                function.
        """
        self._has_mfunc = True
        self.prmts = prmts
        self.mfunc = mfunc
        self.func = MembershipFunc(mfunc, prmts)
        if not self._has_mval:
            self.m = self.func(self.x)

    def set_mval(self, mval):
        self._has_mval = True
        self.x = np.arange(mval.size)
        self.m = mval

    def __call__(self, xV):
        """Return the membership degree for the elements in xV."""
##        assert self.x[0] <= np.asarray(xV).all() <= self.x[-1], \
##               'at least one element of xV does not belong to the universe of discourse.'
        return self.func(xV)

    def __repr__(self):
        return "FuzzySet()"

    def __str__(self):
        if self._has_mfunc:
            return "FuzzySet object\n{}".format(str(self.func))
        else:
            return "FuzzySet object"

    def __len__(self):
        """Return the number of discrete elements of the UoD."""
        return len(self.x)
    
    def _sameuniverse(self, other):
        """Return True if the fuzzy sets are defined over the same UoD,
           False otherwise.
        """
        if len(self) == len(other):
            return (self.x == other.x).all()
        else:
            return False
        
    def __eq__(self, other):
        """Return True if the fuzzy sets are equal, False otherwise."""
        assert self._sameuniverse(other), \
               'fuzzy sets must be defined over the same universe'
        return (self.m == other.m).all()
            

    def __le__(self, other):
        """Return True if the fuzzy set is a subset, False otherwise."""
        assert self._sameuniverse(other), \
               'fuzzy sets must be defined over the same universe'
        return (self.m <= other.m).all()

    def __lt__(self, other):
        """Return True if the fuzzy set is a proper subset, False otherwise."""
        assert self._sameuniverse(other), \
               'fuzzy sets must be defined over the same universe'
        return (self.m < other.m).all()

    def __and__(self, other):
        """Retutn the intersection of two fuzzy sets using the & operator.

            The intersection of two fuzzy sets A and B is a fuzzy set C,
            whose membership function is related to those of A and B by
            μC(x) = min(μA(x), μB(x)) for every x in the universe of discourse
            X. Both A and B must be defined in the same universe of discourse.
        """
        assert self._sameuniverse(other), \
               'fuzzy sets must be defined over the same universe'
        new = self.tnorm(self.m, other.m)
        return FuzzySet(self.x, new)

    def xintersect(self, other, x):
        """Retutn the intersection of two fuzzy sets for the values of x."""
        new = self.tnorm(self.func(x), other.func(x))
        return FuzzySet(x, new)

    def mintersect(self, other):
        """Retutn the intersection of two fuzzy sets for the values of m."""
        #new = self.tnorm(self.m, other.m)
        new = np.minimum(self.m, other.m)
        return FuzzySet(self.x, new)

    def mintersect_ind(self, other, ind):
        """Retutn the intersection of two fuzzy sets for the values of m."""
        new = np.minimum(self.m[ind], other.m[ind])
        return FuzzySet(self.x[ind], new)

    def __or__(self, other):
        """Return the union of two fuzzy sets using the | operator.

            The union of two fuzzy sets A and B is a fuzzy set C,
            whose membership function is related to those of A and B
            by μC(x) = max(μA(x), μB(x)) for every x in the universe
            of discourse X. Both A and B must be defined in the same
            universe of discourse.
        """
        assert self._sameuniverse(other), \
               'fuzzy sets must be defined over the same universe'
        new = self.snorm(self.m, other.m)
        return FuzzySet(self.x, new)
    
    def xunion(self, other, x):
        """Return the union of two fuzzy sets for the values of x."""
        new = self.snorm(self.func(x), other.func(x))
        return FuzzySet(x, new)

    def munion(self, other):
        """Return the union of two fuzzy sets for the values of m."""
        new = self.snorm(self.m, other.m)
        return FuzzySet(self.x, new)
    
    def __invert__(self):
        """Retutn the complement of a fuzzy set using the ~ operator.

            The complement of a fuzzy set A is a fuzzy set B, whose
            membership function is defined by μB(x) = 1 - μA(x) for
            every x in the universe of discourse U. Both A and B must
            be defined in U.
        """
        new = 1 - self.m
        return FuzzySet(self.x, new)

    def complement(self):
        """Helper function. Retutn the complement of a fuzzy set."""
        return ~self

    def cartProduct(self, other):
        """Retutn the Cartesian product of two fuzzy sets.

            The Cartesian product of two fuzzy sets A and B defined over
            X and Y domains respectively is a fuzzy set in the product
            space X*Y with a membership function μ(x) = min(μA(x), μB(x))
        """
        newX, newF = [], []
        for i in range(len(self.x)):
            for j in range(len(other.x)):
                newX.append([self.x[i], other.x[j]])
                newF.append(min(self.m[i], other.m[j]))
        return FuzzySet(newX, newF)
        
    def supp(self):
        """Return the support of a fuzzy set.

            The support of a fuzzy set A is the set of all points x in
            the universe of discourse X such that μ(x) > 0.
        """
        idx = np.where(self.m > 0)
        return np.array(self.x[idx])

    def core(self):
        """Return the core of a fuzzy set.

            The core of a fuzzy set A is the set of all points x in
            the universe of discourse X such that μ(x) = 1.
        """
        idx = np.where(self.m == 1)
        return np.array(self.x[idx])

    def crossover(self):
        """Return the crossover of a fuzzy set.

            The crossover of a fuzzy set A is the set of all points x in
            the universe of discourse X such that μ(x) = 0.5.
        """
        idx = np.where(self.m == 0.5)
        return np.array(self.x[idx])

    def acut(self, a):
        """Returns the a-cut of a fuzzy set.

            The α-cut of a fuzzy set A is a crisp set defined by
            Aα = {x | μ(x) >= α}
        """
        idx = np.where(self.m >= a)
        return np.array(self.x[idx])

    def ascut(self, a):
        """Returns the strong a-cut of a fuzzy set.

            The strong α-cut of a fuzzy set A is a crisp set defined by
            Aα+ = {x | μ(x) > α}
        """
        idx = np.where(self.m > a)
        return np.array(self.x[idx])

    def height(self):
        """Return the height of a fuzzy set."""
        return max(self.m)

    def normalize(self):
        """Normalize the given fuzzy set."""
        self.m /= self.height()
        
    def isnormalized(self):
        """Return True if the fuzzy set is normalized, False otherwise.

            A fuzzy set A is called normal or normalized if its core is
            nonempty.
        """
        return True if self.core else False

    def card(self):
        """Return the cardinality of the fuzzy set.

            The cardinality of a fuzzy set is defined as the sum of the
            membership degrees of all its elements.
        """
        return sum(self.m)

    def relcard(self):
        """Return the cardinality of the fuzzy set.

            The relative cardinality of a fuzzy set is defined as the mean 
            cardinality of the fuzzy set.
        """
        return self.card()/len(self)

    def vagueness(self):
        """Return the vagueness measure of the fuzzy set."""
        idx = np.where(np.logical_and(self.m != 0, self.m != 1))[0]
        a = self.m[idx] * np.log(self.m[idx])
        b = (1-self.m[idx]) * np.log(1-self.m[idx])
        return - np.sum(a+b) / len(self)

    def ambiguity(self):
        """Return the ambiguity measure of the fuzzy set."""
        temp = np.flip(np.sort(self.m))
        idx = np.where(temp != 0)[0]
        now = np.around(temp[idx], 10)
        pval = []
        for j in range(now.shape[0]):
            if j == idx.shape[0]-1:
                p = now[j] * np.log(j+1)
            else:
                p = (now[j] - now[j+1]) * np.log(j+1)
            pval.append(p)
        return sum(pval)
        
    def dsub(self, other):
        """Retutn the degree of subsethood between two fuzzy sets."""
        if self.card():
            return (self.mintersect(other)).card() / self.card()
        else:
            return 0

    def dsub_ind(self, other, ind):
        """Retutn the degree of subsethood between two fuzzy sets."""
        num = np.minimum(self.m[ind], other.m[ind])
        dnm = np.sum(self.m[ind])
        if dnm != 0:
            return np.sum(num)/dnm
        else:
            return 0
  
    def concatenate(self, p):
        """Concentrate a fuzzy set."""
        self.m = self.m ** (1/p)

    def dilate(self, p):
        """Dilate a fuzzy set."""
        self.m = self.m ** (1/p)


    def mult(self, p):
        """Dilate a fuzzy set."""
        self.m = self.m * p 
    
##########
## TODO ##
##########

    
    def distance(self, other):
        """Return the Euclidean distance of two fuzzy sets."""
        s = 0
        for i in range(len(self.m)):
            s += abs((self.m[i] - other.m[i]))**2
        s = s**(1/2)
        return s

    def sim(self, other):
        """Return the Similarity measure of two fuzzy sets."""
        return (self & other).card() / (self | other).card()

    def pos(self, other):
        """Return the Possibility measure of two fuzzy sets.

           Possibility measure describes the degree to which A and B overlap.
        """
        s = []
        for i in range(len(self.m)):
            s.append(min(self.m[i], other.m[i]))
        s = max(s)
        return s

    def nec(self, other):
        """Return the Necessity measure of two fuzzy sets.

           Necessity measure describes the degree to which B is included in A.
        """
        s = []
        for i in range(len(self.m)):
            s.append(max(self.m[i], other.m[i]))
        s = min(s)
        return s

    def cutpoint(self, p):
        newm = []
        for m in self.m:
            if m >= p: newm.append(p)
            else: newm.append(m)
        self.m = np.array(newm)

    @staticmethod
    def Union(list_of_fuzzy):
        f0 = list_of_fuzzy[0]
        for f in list_of_fuzzy[1:]:
            f0 = f0 | f
        return f0
