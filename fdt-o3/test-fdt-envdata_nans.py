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
db = pd.read_csv('pm10db_nans_train.csv')
#db = db.dropna(subset=['Target'])
#db = db.dropna(subset=['pm10_egnatia'])
del db['Year']
del db['Day']
del db['Month']
del db['pm10_egnatia_yest']
del db['WINDD_kord']
idata = db.to_numpy()
X = idata[:, 0:-1]
y = idata[:, -1]
M, N = X.shape
var_names = list(db)[:] 
print("train dataset initialized...")

# Initialize test dataset
db_test = pd.read_csv('pm10db_nans_test.csv')
db_test = db_test.dropna(subset=['Target'])
#db_test = db_test.dropna(subset=['pm10_egnatia'])
del db_test['Year']
del db_test['Day']
del db_test['Month']
del db_test['pm10_egnatia_yest']
del db_test['WINDD_kord']
tdata = db_test.to_numpy()
Xt = tdata[:, 0:-1]
yt = tdata[:, -1]
Mt, Nt = Xt.shape
print("test dataset initialized...")

# Fuzzify train dataset
fname = 'fuzzy_nans_train_73343347_trapmf_custom'
if not os.path.isfile(str(fname) + '.csv'):
    fuzz = Fuzzifier(X, y, task='reg')
    fuzz.get_clusters(init_c=[7,3,3,4,3,3,4,7])
    print(fuzz.best_cntrs)
    fuzz.get_fuzzy_variables(func_type='trapmf', feat_names=var_names)
    fuzz.get_fuzzy_data()
    fdata = fuzz.fuzzy_dataset
    fdata.store_db(fname)
else:
    fdata = FuzzyDataset()
    fdata.load_db(fname)
##    # Change fvar0
##    fdata.fuzzy_variables[0].setmf([('trapmf', [1,1,3,3]),
##                                    ('trapmf', [4,4,6,6]),
##                                    ('trapmf', [7,7,9,9]),
##                                    ('trapmf', [10,10,12,12])])
##    fdata.fuzzy_variables[0].terms = ['Winter', 'Spring', 'Summer', 'Autumn']
##    # Change fvar0
    fdata.fuzzy_variables[0].setmf([('trapmf', [15,20,25,30]),
                                    ('trapmf', [25,30,40,45]),
                                    ('trapmf', [40,45,55,60]),
                                    ('trapmf', [55,60,70,75]),
                                    ('trapmf', [70,75,85,90]),
                                    ('trapmf', [85,90,120,125]),
                                    ('trapmf', [120,130,140,150])])
    # Change fvar4 (temperature)
    fdata.fuzzy_variables[4].setterms(['Low', 'Average', 'High'])
    fdata.fuzzy_variables[4].setmf([('trapmf', [10,10,10,15]),
                                    ('trapmf', [10,15,22,27]),
                                    ('trapmf', [22,27,27,27])])
##    # Change fvar5 (humidity)
##    fdata.fuzzy_variables[5].setterms(['Low', 'Average', 'High'])
##    fdata.fuzzy_variables[5].setmf([('trapmf', [25,25,25,35]),
##                                    ('trapmf', [25,35,55,65]),
##                                    ('trapmf', [55,65,65,65])])
##    # Change fvar6 (windspeed)
##    fdata.fuzzy_variables[6].setterms(['Low', 'Average', 'High', 'Very High'])
##    fdata.fuzzy_variables[6].setmf([('trapmf', [1,1,1,1.5]),
##                                    ('trapmf', [1,1.5,2.5,3]),
##                                    ('trapmf', [2.5,3,4,4.5]),
##                                    ('trapmf', [4,4.5,4.5,4.5])])
    # Change fvar-1
    fdata.fuzzy_variables[-1].setmf([('trapmf', [15,20,25,30]),
                                    ('trapmf', [25,30,40,45]),
                                    ('trapmf', [40,45,55,60]),
                                    ('trapmf', [55,60,70,75]),
                                    ('trapmf', [70,75,85,90]),
                                    ('trapmf', [85,90,120,125]),
                                    ('trapmf', [120,130,140,150])])

### Plot the FuzzyDataset
#FuzzyPlotter(fdata)

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
alpha = 0.51
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
print('## RMSE {:.3f}, {:.6f}'.format(rmse, nrmse[0]))
print('## MAE {:.3f}, {:.6f}'.format(mae, nmae[0]))
# Plot predicted values
linx = np.arange(len(actual_cls))#[0:-1]
fig1 = plt.figure()
plt.plot(linx, yt, 'b', linx, pred, 'r')
plt.xlabel("day")
plt.ylabel("predicted/actual")
plt.title("Fuzzy Decision Tree Regression\nR2 = {:.3f}({:.3f})".format(r2, r2_score(yt,pred)))
plt.show()
fig2 = plt.figure()
plt.scatter(yt, pred)
plt.show()
