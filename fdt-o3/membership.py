# ~~~~~~~~~~~~~
# membership.py
# ~~~~~~~~~~~~~

import numpy as np

class MembershipFunc(object):

    """Membership function generator.

        A membership function for a fuzzy set A on the universe of discourse X
        is defined as µA : X -> [0,1], where each element of X is mapped to a
        value between 0 and 1. This value, called membership value or degree
        of membership, quantifies the grade of membership of the elements of
        X to the fuzzy set A.

        Membership functions allow us to graphically represent a fuzzy set.
        The x axis represents the universe of discourse, whereas the y axis
        represents the degrees of membership in the [0,1] interval.

        Parameters
        ----------
        mfunc : str
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

    names = {'gaussmf' : 'Gaussian membership function',
             'gbellmf' : 'Generalised bell-shaped membership function',
             'sigmf'   : 'Sigmoid membership function',
             'singlemf': 'Singleton membership function',
             'trapmf'  : 'Trapezoidal membership function',
             'trimf'   : 'Triangular membership function',}
        
    def __init__(self, mfunc, prmt):
        self.mfunc = mfunc
        self.prmt = prmt

    def __repr__(self):
        return 'MembershipFunc()'

    def __str__(self):
        return '{}\nParameters: {}\n'.format(self.names[self.mfunc], self.prmt)

    def __call__(self, x):
        """Return the memebership values of x to a fuzzy set with mfucn type
           of membership function and given parameters.

            Parameters
            ----------
            x : 1d array
                Contains the input values, which must lie inside the universe
                of discourse. 
        """
        # Convert input values to array
        if isinstance(x, (int, float)):
            x = np.array([x])
        else:
            try:
                x = np.asarray(x)
            except:
                print('')
        # Call the function
        if self.mfunc is "gaussmf":
            return self.gaussmf(x, self.prmt)
        elif self.mfunc is "gbellmf":
            return self.gbellmf(x, self.prmt)
        elif self.mfunc is "sigmf":
            return self.sigmf(x, self.prmt)
        elif self.mfunc is "singlemf":
            return self.singlemf(x, self.prmt)
        elif self.mfunc is "trapmf":
            return self.trapmf(x, self.prmt)
        elif self.mfunc is "trimf":
            return self.trimf(x, self.prmt)

    def gaussmf(self, x, prmt):
        """Return the Gaussian membership function values for x. 

            Definition of Gaussian function is:
           
                    y(x) = exp(-((x - mean)**2.) / (2 * sigma**2.))
                    
            Inputs
            ------
            x : 1d array
                Independent variable.

            Parameters
            ----------
            mean : float
                Gaussian parameter for center (mean) value.
            sigma : float
                Gaussian parameter for standard deviation.
        """
        # Check for defective parameter values
        assert len(prmt) == 2, \
               'gaussian mf is defined by two parameters; mean and sigma.'
        mean, sigma = prmt
        y = np.exp(-((x - mean)**2.) / (2 * sigma**2.))
        return y
    
    def gbellmf(self, x, prmt):
        """Return the generalised bell-shaped membership function values for x.

           Definition of generalised bell-shaped function is:
           
                    y(x) = 1 / (1 + abs([x - c] / a) ** [2 * b])

            Inputs
            ------
            x : 1d array
                Independent variable.

            Parameters
            ----------
            a : float
                Bell function parameter controlling width.
            b : float
                Bell function parameter controlling slope.
            c : float
                Bell function parameter defining the center.
        """
        # Check for defective parameter values
        assert len(prmt) == 3, \
                ('bell-shaped mf is defined by three parameters; \
                  width, slope and center.')
        a, b, c = prmt
        y = 1 / ( 1 + np.abs( (x-c)/a)**(2*b) )
        return y

    def sigmf(self, x, prmt):
        """Return the sigmoid membership function values for x.

            Definition of sigmoid function is:
            
                    y = 1 / (1. + exp[- c * (x - b)])
                    
            Inputs
            ------
            x : 1d array
                Data vector for independent variable.
                
            Parameters
            ----------
            b : float
                Offset or bias.  This is the center value of the sigmoid, where it
                equals 1/2.
            c : float
                Controls 'width' of the sigmoidal region about `b` (magnitude); also
                which side of the function is open (sign). A positive value of `a`
                means the left side approaches 0.0 while the right side approaches 1.;
                a negative value of `c` means the opposite.
        """
        # Check for defective parameter values
        assert len(prmt) == 2, \
               'sigmoid mf is defined by two parameters; bias and width.'
        b, c = prmt
        y = 1 / ( 1 + np.exp(-c * (x-b)) )
        return y

    def singlemf(self, x, prmt):
        """Return the singleton membership function values for x.

            Inputs
            ------
            x : 1d array
                Independent variable.

            Parameters
            ----------
            a : float
                Defines the x-value, where μ(x) = 1.
        """
        # Check for defective parameter values
        assert len(prmt) == 1, \
               'singleton mf is defined by one parameter [a].'
        y = np.zeros(len(x))
        idx = np.where(x == prmt[0])
        y[idx] = 1
        return y


    def trapmf(self, x, prmt):
        """Return the trapezoidal membership function values for x.

            The parameters {a, b, c, d}, with a <= b <= c <= d determine the
            x coordinates of the four corners of the underlying trapezoidal
            memebership function.

            Inputs
            ------
            x : 1d array
                Independent variable.
        """
        # Check for defective parameter values
        assert len(prmt) == 4, \
               'trapezoidal mf is defined by four parameters [a, b, c, d].'
        
        a, b, c, d = prmt
        
        assert a <= b and b <= c and c <= d, \
               'parameter values must obey a <= b <= c <= d.'
        
        y = np.zeros(len(x))
        
        # Compute left side values (a < x < b)
        idx = np.where(x <= b)[0]
        if a != b:
            y[idx] = self.trimf(x[idx], [a,b,b])
        else:
            y[idx] = self.trimf(x[idx], [b,c,d])
            
        # Compute central values (b <= x <= c)
        idx = np.where(np.logical_and(b <= x, x <= c))[0]
        if b != c:
            y[idx] = 1
        else:
            y[idx] = self.trimf(x[idx], [a,b,d])
        
        # Compute right side values (c < x < d)
        idx = np.where(x >= c)[0]
        if c != d:
            y[idx] = self.trimf(x[idx], [c,c,d])
        else:
            y[idx] = self.trimf(x[idx], [a,b,c])
            
        return y
            
    def trimf(self, x, prmt):
        """Return the triangular membership function values for x.

            The parameters {a, b, c}, with a <= b <= c determine the x coords
            of the three corners of the underlying triangular memebership
            function.
            
            Inputs
            ------
            x : 1d array
                Independent variable.
        """
        # Check for defective parameter values
        assert len(prmt) == 3, \
               'triangular mf is defined by three parameters [a, b, c].'
     
        a, b, c = prmt
        
        assert a <= b and b <= c, \
               'parameter values must obey a <= b <= c.'
        
        y = np.zeros(len(x))

        # Compute left side values (a < x <= b)
        if a != b:
            idx = np.where(np.logical_and(a < x, x <= b))[0]
            y[idx] = (x[idx] - a) / float(b - a)
        else:
            idx = np.where(x <= b)[0]
            y[idx] = 1
 
        # Compute right side values (b <= x < c)
        if b != c:
            idx = np.where(np.logical_and(b <= x, x < c))[0]
            y[idx] = (c - x[idx]) / float(c - b)
        else:
            idx = np.where(x >= b)[0]
            y[idx] = 1
            
        return y


# Testing/Debugging
##from matplotlib import pyplot as plt
##mfunc = 'trapmf'
##prmt = [5,5,5,6]
##x = np.arange(0,10,0.1)
##m = MembershipFunc(mfunc, prmt)
##fig = plt.figure()
##plt.plot(x, m(x))
##plt.show()
