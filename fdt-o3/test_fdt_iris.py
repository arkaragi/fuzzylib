# ~~~~~~~~~~~~~~~~
# test-fdt-iris.py
# ~~~~~~~~~~~~~~~~

import numpy as np
from fdt import FuzzyTree
from fuzzifier import FuzzyDataset
from fuzzyplotter import FuzzyPlotter
from sklearn.datasets import load_iris
   
# Initialize the training dataset
iris = load_iris()
X = iris.data
y = iris.target
M, N = X.shape
# Initialize the fuzzified dataset
fdata = FuzzyDataset()
fdata.load_db('fuzzy-iris')
        
# Plot a Fuzzy Dataset
FuzzyPlotter(fdata)

db_test = [[fdata.fuzzy_dataset[i][j] for i in range(N)] for j in range(M)]

# Build FuzzyTree
alpha = 0.4
tree = FuzzyTree(fdata.fuzzy_dataset, fdata.fuzzy_variables,
                acut=alpha, tlvl=0.85)
tree.build_tree()
tree.print_tree()
r = tree.predict(db_test)
c = tree.to_single_class(r)
acc, correct, wrong = tree.accuracy(c, y)
print('\n#### For acut = {}'.format(alpha))
print('## Model accuracy: {:.3f}'.format(acc))
print('## Number of rules: {}'.format(len(tree.rules)))
print('## Printing fuzzy rules')
tree.print_rules()

# Comparison of FDT's for different a-cuts
model_acc, model_rules = [], []
for a in range(25, 80, 5):
    tree = FuzzyTree(fdata.fuzzy_dataset, fdata.fuzzy_variables,
                 acut=a/100, tlvl=0.9)
    tree.build_tree()
    r = tree.predict(db_test)
    c = tree.to_single_class(r)
    acc, correct, wrong = tree.accuracy(c, y)
    model_acc.append((a, acc))
    model_rules.append((a, tree.rules))
    print('\n#### For acut = {}'.format(a/100))
    print('## Model accuracy: {:.3f}'.format(acc))
    print('## Number of rules: {}'.format(len(tree.rules)))
    print('## Printing fuzzy rules')
    tree.print_rules()
