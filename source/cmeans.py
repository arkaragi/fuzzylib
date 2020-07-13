# ~~~~~~~~~
# cmeans.py
# ~~~~~~~~~
# Author: Aristotelis Karagiannis <arkaragi@physics.auth.gr>

import copy
import numpy as np

class cMeans(object):

    """Fuzzy means clustering algorithm.

        Fuzzy c-means (FCM) is a data clustering technique in which a dataset
        is grouped into N clusters. The major difference with hard clustering
        is that every sample in the dataset can belong to every cluster up to
        a certain degree.

        For example, a point that lies close to the center of a cluster will
        have a high degree of membership in that cluster, and low degrees of
        membership in the other clusters.

        Parameters
        ----------
        data : 2d array, size(S, N)
            Data to be clustered. S is the number of samples and N is the
            number of features within each sample vector.

        c : int, [2, inf)
            The desired number of clusters.

        m : float, (1, inf), default=2
            The weighting exponent m is the hyper-parameter that controls the
            fuzziness of the resulting partition. As m -> 1, the partition
            becomes hard. As m -> inf, the partition becomes completely fuzzy.

        error : float, default=5e-3
            Termination criterion. The algorithm will stop iterating when the
            norm of the difference between U in two successive iterations is
            smaller than the error.

        maxiter : int, default=200
            Maximum number of iterations allowed.

        metric: {'euclidean', 'diagonal', 'mahalanobis'}, default='euclidean'
            The shape of the clusters is determined by the choice of the norm
            inducing matrix A in the distance measure. The Euclidean norm
            induces hypershperical clusters, while the other two induce hyper-
            ellipsoidal clusters. By default it is set to Euclidean.

        fpcoef : {'PC', 'PE', 'XB'}, default='PC'
            The fuzzy partition coefficient is a validity measure for the
            goodness of the obtained partition. It quantifies the separation
            and the compactness of the clusters. By default it is set to the
            classic partition coefficient (PC), although, the Xie-Beni (XB)
            index seem to be more accurate for high dimensional data sets.

        initU : 2d array, size (S, N), default=None
            Initial fuzzy partition matrix. If none provided, it is randomly
            initialized.
  
        seed : int, default=None
            If provided, sets a random seed for the initial partition matrix.
            Mainly for debugging/testing purposes.
    """

    metrics = ['euclidean', 'diagonal', 'mahalanobis']
    partition_coefficients = ['PC', 'PE', 'XB']

    def __init__(self, data, c, m=2, error=5e-3, maxiter=200,
                 metric='euclidean', fpcoef='PC', initU=None, seed=None):

        # Check for defective input values (mainly for debugging)
        assert data.ndim == 2, \
               'input data must be a 2d array of size SxN.'

        assert isinstance(c, int) and (c >= 2), \
               'the number of clusters has to be an integer >= 2.'

        assert isinstance(m, (int, float)) and (m > 1), \
               'the m exponent has to be a real number > 1.'

        assert isinstance(error, (int, float)) and (error > 0), \
               'the error has to be a real number > 0.'

        assert isinstance(maxiter, int) and (maxiter > 0), \
               'maxiter has to be an integer > 0.'

        assert metric in cMeans.metrics, \
               'the norm inducing matrix is not properly defined.'

        assert fpcoef in cMeans.partition_coefficients, \
               'the partition coefficient is not properly defined.'

        assert isinstance(seed, int) or seed==None, \
               'the seed value has to be an integer or None.'

        # Initialize attributes
        self.data = data
        self.S, self.N = data.shape
        self.c = c
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.metric = metric
        self.fpcoef = fpcoef
        self.seed = seed  

        # Initialize a partition matrix
        if initU is not None:
            self.U = copy.deepcopy(initU)
            self.initU = copy.deepcopy(initU)
        else:
            self._initU()
            self.initU = copy.deepcopy(self.U)

        # Execute the algorithm
        self._run()

    def _initU(self):
        """Initialize a random partition matrix."""
        np.random.seed(seed=self.seed)
        t = tuple(1 for i in range(self.c))
        self.U = np.random.dirichlet(t, self.S)

    def _initA(self):
        """Initialize the norm-inducing matrix."""
        if self.metric is 'euclidean':
            self.A = np.identity(self.N)
        elif self.metric is 'diagonal':
            sigma = np.var(self.data, axis=0)
            self.A = np.diag(sigma)
        elif self.metric is 'mahalanobis':
            self.A = np.linalg.inv(np.cov(self.data.T))

    def _run(self):
        """Run the algorithm."""
        self._initA()
        self.no_iter = 0
        while self.no_iter < self.maxiter:
            self._compute_centers()
            self._compute_distances()
            self._compute_functional()
            self._update_partition()
            self.no_iter += 1
            error_now = np.linalg.norm(self.Uold-self.U)
            if error_now < self.error: break
        self._compute_centers()
        self._compute_distances()
        self._compute_functional()
        self._compute_fpc()
        self.J = np.around(np.array(self.J), 9)

    def _compute_centers(self):
        """Compute the cluster centers.

            A cluster center is defined as the mean of all points, weighted by
            their degree of membership to the cluster.
        """
        self.centers = np.empty((self.c, self.N))
        for k in range(self.c):
            to_sum = np.multiply(self.U[:, k:k+1]**self.m, self.data)
            cntr = np.sum(to_sum, axis=0) / np.sum(self.U[:, k:k+1]**self.m)
            self.centers[k, :] = cntr

    def _compute_distances(self):
        """Compute the squared inner-product distance norm."""
        self.D = np.empty((self.S, self.c))
        for k in range(self.c):
            y =  self.data - self.centers[k]
            kth_distances = np.diagonal(y @ self.A @ y.T).T
            self.D[:, k] = np.sqrt(kth_distances)

    def _compute_functional(self):
        """Compute the J-functional."""
        if self.no_iter == 0:
            self.J = []
        j = np.sum((self.U**self.m) * (self.D**2))
        self.J.append(j)

    def _update_partition(self):
        """Update the partition matrix values."""
        self.Uold = copy.deepcopy(self.U)
        for i in range(self.S):
            if (self.D[i] == 0).any():
                self.U[i, :] = 0
                ind0 = np.where(self.D[i] == 0)
                self.U[i, ind0] = 1 / len(ind0)
            else:   
                for k in range(self.c):
                    dnm = 0
                    for j in range(self.c):
                        dnm += (self.D[i][k]/self.D[i][j]) ** (2/(self.m-1))
                    self.U[i, k] = 1 / dnm
            if (self.U[i] < 5e-15).any():
                ind = np.where(self.U[i] < 5e-15)
                self.U[i, ind] = 0
                
    def _compute_fpc(self):
        """Compute the fuzzy partition coefficient."""
        if self.fpcoef is 'PC':
            self.fpc = np.trace(self.U.dot(self.U.T)) / self.S
        elif self.fpcoef is 'PE':
            self.fpc = -np.sum(self.U*np.log2(self.U)) / self.S
        elif self.fpcoef is 'XB':
            self.fpc = 0
            vmin = []
            for k in range(self.c):
                for i in range(self.S):
                    y = self.data[i] - self.centers[k]
                    dst = np.linalg.norm(y)**2
                    self.fpc += self.U[i, k]**self.m * dst
                v = np.linalg.norm(self.centers[k]-self.centers, axis=1)**2
                v = np.delete(v, k)
                vmin.extend(v)
            self.fpc /= (self.S * min(set(vmin)))

    def _predict(self, test):
        """Predict the cluster of a single test item."""
        d = np.empty((1, self.c))
        u = np.empty((1, self.c))
        for k in range(self.c):
            y =  test - self.centers[k]
            d[:, k] =  np.sqrt(y.dot(self.A @ y.T).T)
        for k in range(self.c):
            dnm = 0
            for j in range(self.c):
                dnm += (d[0][k]/d[0][j]) ** (2/(self.m-1))
            u[:, k] = 1 / dnm
        return np.argmax(u)
            
    def predict(self, test_set):
        """Predict a cluster based on the maximum membership degree.

            Parameters
            ----------
            test_set : 2d array, size(M, N)
                pass
        """
        assert test_set.ndim == 2, 'input data must be a 2d array of size SxN.'
        assert test_set.shape[-1] == self.N, 'every element must contain N features.'
        m = len(test_set)
        predicted = np.empty((m, 1))
        for i, test in enumerate(test_set):
            p = self._predict(test)
            predicted[i] = p
        return predicted

# Testing/Debugging (uncomment to run)
######################################
### Initialize the dataset
##from sklearn.preprocessing import MinMaxScaler
##from sklearn.datasets import load_iris
##import skfuzzy as fuzz
##iris = load_iris()
##X = iris.data[:, 2:4]
##scaler = MinMaxScaler()
##scaler.fit(X)
##X = scaler.transform(X)
##y = iris.target
### Comparison between mllib and skfuzzy
##c = 3
##cm = cMeans(X, c, 2, error=5e-3, maxiter=200, seed=0)
##cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
##        X.T, c, 2, error=5e-3, maxiter=200, init=cm.initU.T)
### Print results
##print(cm.J, jm, sep='\n')
##print(np.allclose(cm.U, u.T, 1e-1))
##print(np.allclose(cm.centers, cntr))
##print(cm.no_iter, p)
##print(cm.fpc, fpc)
### Predict class labels
##p = cm.predict(X)
