# ~~~~~~~~~~~~~~~
# test-envdata.py
# ~~~~~~~~~~~~~~~

import copy
import os.path
import numpy as np
import pandas as pd
from fdt import FuzzyTree
from fuzzifier import Fuzzifier, FuzzyDataset
from fuzzyplotter import FuzzyPlotter
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Initialize train dataset
db = pd.read_csv('o3db_train.csv')
db = db.dropna(subset=['Target'])
idata = db.to_numpy()
X = idata[1:, 3:-1]
y = idata[1:, -1]
M, N = X.shape
var_names = list(db)[3:] 
print("train dataset initialized...")

# Initialize test dataset
db_test = pd.read_csv('o3db_test.csv')
db_test = db_test.dropna(subset=['Target'])
tdata = db_test.to_numpy()
Xt = tdata[1:, 3:-1]
yt = tdata[1:, -1]
Mt, Nt = Xt.shape
print("test dataset initialized...")

# Random Forest Regressor
reg = RandomForestRegressor(n_estimators=300)
reg.fit(X,y)
pred = reg.predict(Xt)
rmse = [(a-p)**2for a,p in zip(yt, pred)]
rmse = sum(rmse)/len(rmse)
scaler = MinMaxScaler()
scaler.fit(y.reshape(y.size, 1))
norm_yt = scaler.transform(yt.reshape(yt.size, 1))
norm_pred = scaler.transform(np.array(pred).reshape(yt.size, 1))
ndiff2 = [(a-p)**2 for a,p in zip(norm_yt, norm_pred)]
nrmse = sum(ndiff2)/len(ndiff2)
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
print('## RMSE {:.3f}, {:.6f}'.format(np.sqrt(rmse), np.sqrt(nrmse[0])))
print('## MAE {:.3f}, {:.6f}'.format(mae, nmae[0]))
# Plot predicted values
linx = np.arange(len(yt))
fig1 = plt.figure()
plt.plot(linx, yt, 'b', linx, pred, 'r')
plt.xlabel("day")
plt.ylabel("predicted/actual")
plt.title("Fuzzy Decision Tree Regression\nR2 = {:.3f}({:.3f})".format(r2, r2_score(yt,pred)))
plt.show()
fig2 = plt.figure()
plt.scatter(yt, pred)
plt.show()
