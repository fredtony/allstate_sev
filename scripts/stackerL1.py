# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 08:19:34 2016

@author: Tony
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import copy
from numba import jit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import gmean
from scipy.optimize import minimize
import xgboost as xgb
#from joblib import Parallel, delayed
#import multiprocessing
import cPickle
import tqdm
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor

shift = 200

@jit
def fair_obj(preds, dtrain, fair_constant=0.7):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = fair_constant / (abs(x) + fair_constant)
    grad = den * x
    hess = den * den
    return grad, hess

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(y, yhat)

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

MODELDIR = '../L1/'
current_model_list = os.listdir(MODELDIR)
len_models = len(current_model_list)/2
oof_list = current_model_list[:len_models]
preds_list = current_model_list[len_models:]
train_ids = pd.read_csv(MODELDIR+oof_list[1], usecols=['id'], squeeze=True).values
test_ids = pd.read_csv(MODELDIR+preds_list[1], usecols=['id'], squeeze=True).values

def out_from_preds(train_ids, cv_pred, test_ids, test_pred, folds, score, clf_type):
    now = datetime.now()
    
    pred = pd.DataFrame({'id': test_ids, 'loss': pd.Series(test_pred)})
    pred_file = '../L2/L2ensemble_submission_{}fold-{}-{:.4f}-{}.csv.gz'.format(
        folds, clf_type, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing test probabilities: {}".format(pred_file))
    pred.to_csv(pred_file, index=False, compression='gzip')
    
    oof_pred = pd.DataFrame({'id': train_ids, 'loss': pd.Series(cv_pred)})
    oof_pred_file = '../L2/L2ensemble_oof_preds_{}fold-{}-{:.4f}-{}.csv.gz'.format(
        folds, clf_type, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing oof probabilities: {}".format(oof_pred_file))
    oof_pred.to_csv(oof_pred_file, index=False, compression='gzip')
    return

def import_models(model_list):
    model = pd.read_csv(MODELDIR+model_list[0])
    for elem in model_list[1:]:
        next_model = pd.read_csv(MODELDIR+elem)
        model = model.merge(next_model, on='id', how='left')
    return model
    
train = import_models(oof_list)
train = train.merge(pd.read_csv('../input/train.csv', usecols=['id','loss']),
                    on='id', how='left')
y = train.iloc[:,-1].values
train = train.iloc[:,1:-1].values
test = import_models(preds_list).values[:,1:]

def eval_slsqp(train_cv, y, test, train_ids, test_ids, nfolds=10, random_state=0):
    skf = kfold_regression(y, nfolds, random_state=random_state)
    test_pred = np.zeros((test.shape[0], nfolds))
    cv_pred = np.zeros(y.shape)
    minimizers = []
    scores = np.zeros(nfolds)
    for i, (train_idx, cv_idx) in enumerate(skf):
        train, y_train = train_cv[train_idx,:], y[train_idx]
        cv, y_cv = train_cv[cv_idx,:], y[cv_idx]
        minimizers.append(minimize_weights_mae(train, y_train))
        cv_pred[cv_idx] = np.dot(cv, minimizers[i].x)
        scores[i] = mean_absolute_error(y_cv, cv_pred[cv_idx])
        print("Fold {} MAE: {}".format(i+1, scores[i]))
        test_pred[:,i] = np.dot(test, minimizers[i].x)
    del train, cv
 
    test_pred = np.average(test_pred, 1)
    
    score = mean_absolute_error(y, cv_pred)
    print("Out of fold (predicted LB) CV MAE: {}".format(score))
    
    out_from_preds(train_ids, cv_pred, test_ids, test_pred, nfolds, score, 'slsqp')
eval_slsqp(train, y, test, train_ids, test_ids, 10, 181)

def eval_clf(clf, train_cv, y, test, train_ids, test_ids, clf_type='clf', nfolds=10, random_state=0):
    skf = kfold_regression(y, nfolds, random_state=random_state)
    test_pred = np.zeros((test.shape[0], nfolds))
    cv_pred = np.zeros(y.shape)
    clfs = []
    scores = np.zeros(nfolds)
    for i, (train_idx, cv_idx) in enumerate(skf):
        clfs.append(copy.deepcopy(clf))
        train, y_train = train_cv[train_idx,:], y[train_idx]
        cv, y_cv = train_cv[cv_idx,:], y[cv_idx]
        clfs[i].fit(train, y_train)
        cv_pred[cv_idx] = clfs[i].predict(cv)
        scores[i] = mean_absolute_error(y_cv, cv_pred[cv_idx])
        print("Fold {} MAE: {}".format(i+1, scores[i]))
        test_pred[:,i] = clfs[i].predict(test)
    del train, cv
 
    test_pred = np.average(test_pred, 1)
    
    score = mean_absolute_error(y, cv_pred)
    print("Out of fold (predicted LB) CV MAE: {}".format(score))
    
    out_from_preds(train_ids, cv_pred, test_ids, test_pred, nfolds, score, clf_type)
                   
#clf = LinearRegression(fit_intercept=False)
#clf = RandomForestRegressor(n_estimators=10, criterion='mae',random_state=0)
#clf = ExtraTreesRegressor(n_estimators=40, criterion='mae', random_state=0)
clf = GradientBoostingRegressor(loss='lad')
#clf = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion='mae'),
#                         loss='linear', random_state=0)
#clf = AdaBoostRegressor(loss='linear', random_state=0)
#clf = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
#clf = SGDRegressor(loss='huber', penalty='none', random_state=0, fit_intercept=False, n_iter=1000, learning_rate='optimal')
#clf = MLPRegressor(hidden_layer_sizes=(100,20), random_state=0,
#                    tol=.0001, verbose=True)
eval_clf(clf, train, y, test, train_ids, test_ids, 'GBR', 10, 156)

def eval_xgb(train, y, test, train_ids, test_ids, nfolds=5, random_state=0):
    xgb_params = {
    "base_score": np.median(y),
    "booster": "gblinear",
    "objective": "reg:linear",
    "max_depth": 5,
    "eta": 0.05,
    "silent": 1,
    }
    num_round = 10000
    early_stopping_rounds = 30
    #nfolds = nfolds
    #watchlist = [(dtrain,'train'), (dcv,'cv')]
    #bst = xgb.train(xgb_params, dtrain, num_round, watchlist, maximize=True)
    
    skf = kfold_regression(y, nfolds, random_state=random_state)
    test_pred = np.zeros((test.shape[0], nfolds))
    cv_pred = np.zeros(y.shape)
    xgb_models = []
    scores = np.zeros(nfolds)
    dtest = xgb.DMatrix(test)
    for i, (train_idx, cv_idx) in enumerate(skf):
        dtrain = xgb.DMatrix(train[train_idx,:], y[train_idx])
        dcv = xgb.DMatrix(train[cv_idx,:], y[cv_idx])
        watchlist = [(dtrain,'train'), (dcv,'cv')]
        bst = xgb.train(xgb_params, dtrain, num_round, watchlist, obj=fair_obj,
            feval=xg_eval_mae, verbose_eval=25, early_stopping_rounds=early_stopping_rounds)
        cv_pred[cv_idx] = bst.predict(dcv)#, ntree_limit=bst.best_ntree_limit)
        scores[i] = mean_absolute_error(y[cv_idx], cv_pred[cv_idx])
        print("Fold {} MAE: {}".format(i+1, scores[i]))
        test_pred[:,i] = bst.predict(dtest)#, ntree_limit=bst.best_ntree_limit)
        xgb_models.append(bst)
    del dtrain
    del dcv
    
    #y_pred = np.zeros((1183748, len(skf)))
    #for i in xrange(len(skf)):
    #    y_pred[:,i] = xgb_models[i].predict(dtest)
    test_pred = test_pred.mean(1)
    
    score = mean_absolute_error(y, cv_pred)
    print("Out of fold (predicted LB) CV MAE: {}".format(score))
    
    out_from_preds(train_ids, cv_pred, test_ids, test_pred, nfolds, score, 'xgb')
    return

eval_xgb(train, y, test, train_ids, test_ids, nfolds=10, random_state=5)