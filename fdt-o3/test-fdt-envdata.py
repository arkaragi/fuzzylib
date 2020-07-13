# ~~~~~~~~~~~~~~~
# test-envdata.py
# ~~~~~~~~~~~~~~~

import copy
import time
import os.path
import numpy as np
import pandas as pd
from fdt import FuzzyTree
from fuzzifier import Fuzzifier, FuzzyDataset
from fuzzyplotter import FuzzyPlotter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Initialize train dataset
db = pd.read_csv('o3db_train.csv')
db = db.dropna(subset=['Target'])
del db['Year']
del db['Day']
del db['Month']
del db['WINDD_kord']
idata = db.to_numpy()
X = idata[:, 0:-1]
y = idata[:, -1]
M, N = X.shape
var_names = list(db)[:] 
print("train dataset initialized...")

# Initialize test dataset
db_test = pd.read_csv('o3db_test.csv')
db_test = db_test.dropna(subset=['Target'])
del db_test['Year']
del db_test['Day']
del db_test['Month']
del db_test['WINDD_kord']
tdata = db_test.to_numpy()
Xt = tdata[:, 0:-1]
yt = tdata[:, -1]
Mt, Nt = Xt.shape
print("test dataset initialized...")

# Fuzzify train dataset
fname = 'fuzzy_train_7333333347_trapmf_custom'
if not os.path.isfile(str(fname) + '.csv'):
    fuzz = Fuzzifier(X, y, task='reg')
    fuzz.get_clusters(init_c=[7,3,3,3,3,3,3,3,4,7])
    print(fuzz.best_cntrs)
    fuzz.get_fuzzy_variables(func_type='trapmf', feat_names=var_names)
    fuzz.get_fuzzy_data()
    fdata = fuzz.fuzzy_dataset
    fdata.store_db(fname)
else:
    fdata = FuzzyDataset()
    fdata.load_db(fname)
    # Change fvar0
##    fdata.fuzzy_variables[0].setmf([('trapmf', [15,15,15,20]),
##                                    ('trapmf', [15,20,22,25]),
##                                    ('trapmf', [22,25,27,30]),
##                                    ('trapmf', [27,30,32,35]),
##                                    ('trapmf', [32,35,37,40]),
##                                    ('trapmf', [37,40,42,45]),
##                                    ('trapmf', [42,45,60,80])])
    # Change fvar1
    fdata.fuzzy_variables[1].setmf([('trapmf', [20,20,20,30]),
                                    ('trapmf', [20,30,35,50]),
                                    ('trapmf', [35,50,50,50])])   
    # Change fvar2
    fdata.fuzzy_variables[2].setmf([('trapmf', [20,20,20,30]),
                                    ('trapmf', [20,30,35,50]),
                                    ('trapmf', [35,50,50,50])])  
    # Change fvar3
    fdata.fuzzy_variables[3].setmf([('trapmf', [20,20,20,30]),
                                    ('trapmf', [20,30,35,50]),
                                    ('trapmf', [35,50,50,50])]) 
    # Change fvar4
    fdata.fuzzy_variables[4].setmf([('trapmf', [70,70,70,90]),
                                    ('trapmf', [70,90,100,120]),
                                    ('trapmf', [100,120,120,120])])
    # Change fvar5
    fdata.fuzzy_variables[5].setmf([('trapmf', [70,70,70,90]),
                                    ('trapmf', [70,90,100,120]),
                                    ('trapmf', [100,120,120,120])])
    # Change fvar6
    fdata.fuzzy_variables[6].setmf([('trapmf', [5,5,5,15]),
                                    ('trapmf', [5,15,20,35]),
                                    ('trapmf', [20,35,35,35])])
    # Change fvar7
    fdata.fuzzy_variables[7].setmf([('trapmf', [40,40,40,50]),
                                    ('trapmf', [40,50,60,70]),
                                    ('trapmf', [60,70,70,70])])
    # Change fvar8
    fdata.fuzzy_variables[8].setmf([('trapmf', [0.5,0.5,0.5,1]),
                                    ('trapmf', [0.5,1,1.5,2]),
                                    ('trapmf', [1.5,2,2.5,3]),
                                    ('trapmf', [2.5,3,3,3])])
    # Change fvar-1
##    fdata.fuzzy_variables[-1].setmf([('trapmf', [15,15,15,20]),
##                                    ('trapmf', [15,20,22,25]),
##                                    ('trapmf', [22,25,27,30]),
##                                    ('trapmf', [27,30,32,35]),
##                                    ('trapmf', [32,35,37,40]),
##                                    ('trapmf', [37,40,42,45]),
##                                    ('trapmf', [42,45,60,80])])
    fdata.fuzzy_variables[-1].terms = ['Very Low', 'Low', 'Below Average',
                                       'Average', 'Above Average', 'High', 'Very High']

### Plot the FuzzyDataset
##FuzzyPlotter(fdata)

# Fuzzify test dataset
fdb_test = []
for i, row in enumerate(Xt):
    temp = []
    for j, fvar in fdata.fuzzy_variables.items():
        dummy = []
        if j == -1: break
        for fuzz in fvar.fuzzy:
            if  np.isnan(row[j]):
                s = 'np.ones(len(fvar.fuzzy))'
            else:
                s = 'fuzz.func.{}(np.array([row[j]]), fuzz.prmts)'.format(fuzz.mfunc)
            dummy.extend(eval(s))
        temp.append(dummy)
    fdb_test.append(temp)
fdb_test = np.array(fdb_test)
# Find max class membership value
actual_cls = []
for i in range(Mt):
    temp = []
    for f in fdata.fuzzy_variables[-1].fuzzy:
        s = 'f.func.{}(np.array([yt[i]]), f.prmts)'.format(f.mfunc)
        temp.append(eval(s))
    actual_cls.append(np.argmax(temp))

# Build and print FuzzyTree
alpha = 0.01
beta = 0.65
fd = copy.deepcopy(fdata.fuzzy_dataset)
fv = copy.deepcopy(fdata.fuzzy_variables)
fdt = FuzzyTree(fd, fv, acut=alpha, tlvl=beta, maxdepth=None)
fdt.build_tree()
##fdt.print_rules()
# Predict class labels
r = fdt.predict(fdb_test)
c = fdt.to_single_class(r)
acc, correct, wrong = fdt.accuracy(c, actual_cls)
# Print results
print('\n#### For acut = {}, tlvl = {}, maxdepth={}'.format(alpha, beta, fdt.maxdepth))
print('## Dataset: {}'.format(fname))
print('## Number of rules: {}'.format(len(fdt.rules)))
print('## Model classification accuracy: {:.3f}'.format(acc))

# Predict regressed values
pred, rmse = fdt.regress(fdb_test, yt)
pred = np.array(pred)

##pred = pred[1:]
##yt = yt[0:-1]

mae = sum([abs(a-p) for a,p in zip(yt, pred)])/len(pred)
r2 = 1 - sum([(t-p)**2 for t,p in zip(yt,pred)])/sum([(t-yt.mean())**2 for t in yt])
scaler = MinMaxScaler()
scaler.fit(y.reshape(y.size, 1))
norm_yt = scaler.transform(yt.reshape(yt.size, 1))
norm_pred = scaler.transform(np.array(pred).reshape(yt.size, 1))
ndiffa = [abs(a-p) for a,p in zip(norm_yt, norm_pred)]
ndiff2 = [(a-p)**2 for a,p in zip(norm_yt, norm_pred)]
nrmse = sum(ndiff2)/len(ndiff2)
nmae = sum(ndiffa)/len(ndiffa)
# Print results
print('## Model regression accuracy: R2 = {:.3f}({:.3f})'.format(r2, r2_score(yt,pred)))
print('## RMSE {:.3f}, {:.4f}'.format(np.sqrt(rmse), np.sqrt(nrmse[0])))
print('## MAE {:.3f}, {:.6f}'.format(mae, nmae[0]))
# Plot predicted values
linx = np.arange(len(actual_cls))#[0:-1]
fig1 = plt.figure()
plt.plot(linx, yt, 'b', label='actual')
plt.plot(linx, pred, 'r', label='predicted')
plt.xlabel("Day of the year 2013")
plt.ylabel("PM10 concentration (μg/m^3)")
plt.title("Timeseries of O3 concentration (2013)")
plt.legend()
plt.show()
fig2 = plt.figure()
plt.scatter(yt, pred)
xy = np.arange(0,80)
plt.plot(xy, xy, 'k')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted")
plt.show()
