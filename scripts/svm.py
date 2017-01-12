# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 14:34:40 2016

@author: Tony
"""

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from datetime import datetime
from tqdm import trange
from scipy.optimize import minimize
import scipy.sparse as sps
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def kfold_regression(y, n_folds, random_state=0):
    np.random.seed(random_state)
    kf_list = [[] for x in xrange(n_folds)]
    len_folds = y.shape[0] // n_folds
    extra = y.shape[0] % n_folds
    order = y.argsort()
    for i in xrange(y.shape[0]//n_folds):
        i = i*n_folds
        order[i:i+n_folds] = np.random.permutation(order[i:i+n_folds])
    for i in xrange(n_folds):
        for j in xrange(len_folds):
            kf_list[i].append(order[j*n_folds+i])
    if extra != 0:
        extras = np.random.choice(n_folds, extra, replace=False)
        for i, fold in enumerate(extras):
            kf_list[fold].append(order[n_folds*len_folds+i])
    train_test_tuples = []
    for i in xrange(n_folds):
        train_index = []
        for j, indices in enumerate(kf_list):
            if i != j:
                train_index += indices
        train_test_tuples.append((train_index, kf_list[i]))
    return train_test_tuples

def weight_fn(w, X, y):
    return mean_absolute_error(y, np.dot(X, w))

def weight_fn_deriv(w, X, y):
    signs = np.sign(np.dot(X, w) - y).reshape((X.shape[0], 1))
    return np.multiply(signs, X).mean(0)
    
def minimize_weights_mae(X, y):
    n = X.shape[1]
    guess = np.ones(n) / float(n)
    cons = ({'type': 'eq',
             'fun' : lambda w: w.sum() - 1,
             'jac' : lambda w: np.ones_like(w)},)
    bounds = []
    for i in range(n):
        bounds.append((0,1)) 
    return minimize(weight_fn, guess, args=(X,y), jac=weight_fn_deriv, bounds=bounds,
                    constraints = cons, method='SLSQP', options={'disp': True})

## read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

f_cat = np.array([f for f in train.columns if 'cat' in f])
f_prob_ordered = np.array(['cat88', 'cat99', 'cat101', 'cat102', 'cat107', 
                'cat109', 'cat110', 'cat111', 'cat113', 'cat114', 'cat115'])
f_prob_unordered = f_cat[np.in1d(f_cat, f_prob_ordered, invert=True)]
f_num = np.array([f for f in train.columns if 'cont' in f])

index = list(train.index)
print index[0:10]
np.random.shuffle(index)
print index[0:10]
train = train.iloc[index]
'train = train.iloc[np.random.permutation(len(train))]'

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
shift = 430
y = np.log(train['loss']+shift).values
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

## Categorical data
for f in f_prob_unordered:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    sparse_data.append(csr_matrix(dummy))

sparse_data.append(csr_matrix(minmax_scale(tr_te[f_prob_ordered])))

## Continuous data
for f in f_num:
    cut50 = pd.cut(tr_te[f], 50, labels=False)
    cut100 = pd.cut(tr_te[f], 100, labels=False)
    sparse_data.append(csr_matrix(pd.get_dummies(cut50)))
    sparse_data.append(csr_matrix(pd.get_dummies(cut100)))
#scaler = StandardScaler()
#sparse_data.append(csr_matrix(scaler.fit_transform(tr_te[f_num])))

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data)

## cv-folds
nfolds = 10
folds = kfold_regression(y, nfolds, random_state=0)

## train models
pred_oof = np.zeros(xtrain.shape[0])
pred_test = np.zeros((xtest.shape[0], nfolds))

for i, (inTr, inTe) in enumerate(folds):
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    clf = []
    clf.append(SVR())
    clf[i].fit(xtr, ytr)
    pred_oof[inTe] = np.exp(clf[i].predict(xte))-shift
    pred_test[:,i] = np.exp(clf[i].predict(xtest))-shift
    score = mean_absolute_error(np.exp(yte)-shift, pred_oof[inTe])
    print('Fold {} - MAE: {}'.format(i+1, score))

#score = mean_absolute_error(np.exp(y)-shift, pred_oof)
score = mean_absolute_error(y, pred_oof)
score = str(round(score, 5))
print('Total - MAE:', score)


now = datetime.now()

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oof})
df.to_csv('../L0/oof_svr_{}fold_{}_{}.csv.gz'\
              .format(nfolds, score, now.strftime("%Y-%m-%d-%H-%M")),
          index = False, compression='gzip')

## test predictions
df = pd.DataFrame({'id': id_test, 'loss': pred_test.mean(1)})
df.to_csv('../L0/submission_svr_{}fold_{}_{}.csv.gz'\
              .format(nfolds, score, now.strftime("%Y-%m-%d-%H-%M")),
          index = False, compression='gzip')