# ~~~~~~~~~~~~~~~
# test-fuzzify.py
# ~~~~~~~~~~~~~~~

import numpy as np
from fuzzifier import Fuzzifier, FuzzyDataset
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
  
# Initialize the Iris dataset
print('####Fuzzifing the Iris dataset...')
iris = load_iris()
X = iris.data
y = iris.target
M, N = X.shape 
var_names = iris.feature_names
# Fuzzify and export
fuzz = Fuzzifier(X, y)
fuzz.get_clusters(best_c=[3,4])
fuzz.get_fuzzy_variables(feat_names=var_names)
fuzz.get_fuzzy_data()
fdata = fuzz.fuzzy_dataset
fdata.store_db('fuzzy-iris')

# Initialize the Wine dataset
print('\n####Fuzzifing the Wine dataset...')
wine = load_wine()
X = wine.data
y = wine.target
M, N = X.shape 
var_names = wine.feature_names
# Fuzzify and export
fuzz = Fuzzifier(X, y)
fuzz.get_clusters(best_c=[3,4])
fuzz.get_fuzzy_variables(feat_names=var_names)
fuzz.get_fuzzy_data()
fdata = fuzz.fuzzy_dataset
fdata.store_db('fuzzy-wine')

# Initialize the Breast Cancer dataset
print('\n####Fuzzifing the Breast Cancer dataset...')
canc = load_breast_cancer()
X = canc.data
y = canc.target
M, N = X.shape 
var_names = canc.feature_names
# Fuzzify and export
fuzz = Fuzzifier(X, y)
fuzz.get_clusters(best_c=[3,4])
fuzz.get_fuzzy_variables(feat_names=var_names)
fuzz.get_fuzzy_data()
fdata = fuzz.fuzzy_dataset
fdata.store_db('fuzzy-canc')

### Initialize the Digits dataset
##print('\n####Fuzzifing the Breast Cancer dataset...')
##dig = load_digits()
##X = dig.data
##y = dig.target
##M, N = X.shape 
##var_names = dig.feature_names
### Fuzzify and export
##fuzz = Fuzzifier(X, y)
##fuzz.get_clusters(best_c=[3,4])
##fuzz.get_fuzzy_variables(feat_names=var_names)
##fuzz.get_fuzzy_data()
##fdata = fuzz.fuzzy_dataset
##fdata.store_db('fuzzy-dig')
