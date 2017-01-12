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
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

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
prob_unordered = f_cat[np.in1d(f_cat, f_prob_ordered, invert=True)]
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
y = np.log(train['loss'].values + shift)
y = train['loss'].values
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)


scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

## cv-folds
nfolds = 10
#folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)
folds = kfold_regression(y, nfolds, random_state=0)

## train models
nbags = 10
nepochs = 55
pred_oob = np.zeros((xtrain.shape[0], nbags))
pred_oof = np.zeros(xtrain.shape[0])
pred_test_oob = np.zeros((xtest.shape[0], nfolds, nbags))
pred_test = np.zeros((xtest.shape[0], nfolds))
minimizers = []

for i, (inTr, inTe) in enumerate(folds):
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    for j in range(nbags):
        model = nn_model()
        fit = model.fit(xtr, ytr, batch_size=128, shuffle=True,
                        nb_epoch = nepochs, verbose = 0)
#        pred_oob[inTe,j] = (np.exp(model.predict(xte, batch_size=800))-shift)\
#                                .reshape(len(inTe))
        pred_oob[inTe,j] = (model.predict(xte, batch_size=800)).reshape(len(inTe))
#        pred_test_oob[:,i,j] = (np.exp(model.predict(xtest, batch_size=800))-shift)\
#                                .reshape((xtest.shape[0],))
        pred_test_oob[:,i,j] = (model.predict(xtest, batch_size=800)).reshape((xtest.shape[0],))
#        score = mean_absolute_error(np.exp(yte)-shift, pred_oob[inTe,j])
        score = mean_absolute_error(yte, pred_oob[inTe,j])
        print('Bag {} - MAE: {}'.format(j+1, score))
#    fold_minimizer = minimize_weights_mae(pred_oob[inTe,:], np.exp(yte)-shift)
    fold_minimizer = minimize_weights_mae(pred_oob[inTe,:], yte)
    minimizers.append(fold_minimizer)
    pred_oof[inTe] = np.dot(pred_oob[inTe,:], fold_minimizer.x)
    pred_test[:,i] = np.dot(pred_test_oob[:,i,:], fold_minimizer.x)
#    score = mean_absolute_error(np.exp(yte)-shift, pred_oof[inTe])
    score = mean_absolute_error(yte, pred_oof[inTe])
    print('Fold {} - MAE: {}'.format(i+1, score))

#score = mean_absolute_error(np.exp(y)-shift, pred_oof)
score = mean_absolute_error(y, pred_oof)
score = str(round(score, 5))
print('Total - MAE:', score)


now = datetime.now()

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oof})
df.to_csv('../L0/oof_nn_{}fold_{}bag_{}_{}.csv.gz'\
              .format(nfolds, nbags, score, now.strftime("%Y-%m-%d-%H-%M")),
          index = False, compression='gzip')

## test predictions
df = pd.DataFrame({'id': id_test, 'loss': pred_test.mean(1)})
df.to_csv('../L0/submission_nn_{}fold_{}bag_{}_{}.csv.gz'\
              .format(nfolds, nbags, score, now.strftime("%Y-%m-%d-%H-%M")),
          index = False, compression='gzip')