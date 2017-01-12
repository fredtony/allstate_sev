# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 13:22:08 2016

@author: Tony
"""

import numpy as np
import pandas as pd
from numba import jit
import lightgbm as lgb

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import string
from scipy.optimize import minimize


shift = 420
SEED = 2016

COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
               'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
               'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
               'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')
ohe_list = ['cat73', 'cat93', 'cat112']

def exp0(x, shift=shift):
    return np.exp(x)-shift

def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r

def convert_cat_to_num(data):
    translation = list(string.ascii_uppercase)[:-1]
    for elem_i in translation[:13]:
        for elem_j in translation[:25]:
            translation.append(elem_i + elem_j)
    for i in range(len(translation)):
        data.replace(translation[i], i, inplace=True)
    data.replace('ZZ', -1, inplace=True)
    return

@jit
def fair_obj(preds, train_data, fair_constant=1.7):
    labels = train_data.get_label()
    x = (preds - labels)
    den = fair_constant / (np.abs(x) + fair_constant)
    grad = den * x
    hess = den * den
    return grad, hess

def xg_eval_mae(preds, train_data):
    labels = train_data.get_label()
#    return 'mae', mean_absolute_error(labels, preds), False
    return 'mae', mean_absolute_error(exp0(preds), exp0(labels)), False
    
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
                                      
def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain

def filter_cat(x, remove):
    if x in remove:
        return np.nan
    return x

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
        

if __name__ == "__main__":

    print('\nStarted')
    directory = '../input/'
    train = pd.read_csv(directory + 'train.csv')
    test = pd.read_csv(directory + 'test.csv')

    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    categorical_feats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test, ntrain = mungeskewed(train, test, numeric_feats)
    
    # taken from Vladimir's script (https://www.kaggle.com/iglovikov/allstate-claims-severity/xgb-1114)
    for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)

            train_test[column] = train_test[column].apply(lambda x: filter_cat(x, remove), 1)

    # taken from Ali's script (https://www.kaggle.com/aliajouz/allstate-claims-severity/singel-model-lb-1117)
    train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
    train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
    train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
    train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
    train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
    train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
    train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

    train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"]) + 0000.1)
    train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"]) + 0000.1)
    train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"]) + 0000.1)
    train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"]) + 0000.1)
    train_test["cont14"] = (np.maximum(train_test["cont14"] - 0.179722, 0) / 0.665122) ** 0.25

    print('')
    for comb in itertools.combinations(COMB_FEATURE, 2):
        feat = comb[0] + "_" + comb[1]
        train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
        train_test[feat] = train_test[feat].apply(encode)
        print('Combining Columns:', feat)

    print('Encoding categorical variables')
    convert_cat_to_num(train_test)
#    for col in categorical_feats:
#        print('Analyzing Column:', col)
#        train_test[col] = train_test[col].apply(encode)

#    print(train_test[categorical_feats])
    
    train_test = pd.get_dummies(train_test, columns=ohe_list)
    
    for i in [1,3,5,6,7,9,10,11,12,13]:
        col = 'cont' + str(i)
        min_val = (train_test[col]==train_test[col].min())
        train_test[col+'min'] = min_val.astype('int')
        train_test.loc[train_test[col]==min_val,col] = np.nan
    col = 'cont3'
    max_val = train_test[col]==train_test[col].max()
    train_test['cont3max'] = max_val.astype('int')
    train_test.loc[train_test[col]==max_val, col] = np.nan 
    
    ss = StandardScaler()
    train_test[numeric_feats] = \
        ss.fit_transform(train_test[numeric_feats].values)

    train = train_test.iloc[:ntrain, :].copy()
    test = train_test.iloc[ntrain:, :].copy()
    del train_test

    print('Median Loss:', train.loss.median())
    print('Mean Loss:', train.loss.mean())

    test_ids = pd.read_csv('../input/test.csv', usecols=['id'], squeeze=True)
    train_ids = pd.read_csv('../input/train.csv', usecols=['id'], squeeze=True)
    train_y = np.log(train['loss'] + shift)
    train_x = train.drop(['loss','id'], axis=1)
    test_x = test.drop(['loss','id'], axis=1)

    del test
    
#    cv_preds = np.zeros(train_ids.shape[0])
#    y_preds = np.zeros((test_ids.shape[0], n_folds))
    
    n_folds = 10
    nbags = 3
    pred_oob = np.zeros((train_x.shape[0], nbags))
    pred_oof = np.zeros(train_x.shape[0])
    pred_test_oob = np.zeros((test_x.shape[0], n_folds, nbags))
    pred_test = np.zeros((test_x.shape[0], n_folds))
    minimizers = []

    kf = kfold_regression(train_y, n_folds, random_state=37)
    for i, (train_index, cv_index) in enumerate(kf):
        print('\n Fold %d' % (i+1))
        X_train, X_cv = train_x.iloc[train_index], train_x.iloc[cv_index]
        y_train, y_cv = train_y.iloc[train_index], train_y.iloc[cv_index]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_cv = lgb.Dataset(X_cv, y_cv, reference=lgb_train)
        
        for j in range(nbags):
            params = {
                'seed': j+i*15,
                'application':'regression',
#                'metric': 'l1',
                'colsample_bytree': 0.7,
                'silent': 1,
                'subsample': 0.7,
                'learning_rate': 0.02,
                'num_leaves': 200,
                'lambda_l1': 1.0,
                'min_hessian': 100,
                'max_depth': 12,
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'boosting': 'gbdt'}
                
            gbm = lgb.train(params, lgb_train, num_boost_round=100000, #fobj=fair_obj,
                valid_sets=[lgb_train,lgb_cv], verbose_eval=100, feval=xg_eval_mae,
                early_stopping_rounds=100, feature_name=train_x.columns.tolist())
            
            pred_oob[cv_index,j] = (exp0(gbm.predict(X_cv, num_iteration=gbm.best_iteration)))\
                                    .reshape(len(cv_index))
            pred_test_oob[:,i,j] = (exp0(gbm.predict(test_x, num_iteration=gbm.best_iteration)))\
                                    .reshape((test_x.shape[0],))
            score = mean_absolute_error(exp0(y_cv), pred_oob[cv_index,j])
            print('Bag {} - MAE: {}'.format(j+1, score))
        fold_minimizer = minimize_weights_mae(pred_oob[cv_index,:], exp0(y_cv))
        minimizers.append(fold_minimizer)
        pred_oof[cv_index] = np.dot(pred_oob[cv_index,:], fold_minimizer.x)
        pred_test[:,i] = np.dot(pred_test_oob[:,i,:], fold_minimizer.x)
        score = mean_absolute_error(exp0(y_cv), pred_oof[cv_index])
        print('Fold {} - MAE: {}'.format(i+1, score))
    
    #score = mean_absolute_error(np.exp(y)-shift, pred_oof)
    score = mean_absolute_error(exp0(train_y), pred_oof)
    score = str(round(score, 5))
    print('Total - MAE:', score)

    print("Writing test results")
    result = pd.DataFrame(pred_test.mean(1), columns=['loss'])
    result["id"] = test_ids
    result = result.set_index("id")

    now = datetime.now()
    sub_file = '../L0/submission_{}fold_{}bag-lgbm_{}_{}.csv.gz'\
        .format(n_folds, nbags, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='id', compression='gzip')
    
    print("Writing cv results")
    result_cv = pd.DataFrame(pred_oof, columns=['loss'])
    result_cv["id"] = train_ids
    result_cv = result_cv.set_index("id")

    sub_file_cv = '../L0/oof_{}fold_{}bag-lgbm_{}_{}.csv.gz'\
        .format(n_folds, nbags, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("Writing submission: %s" % sub_file_cv)
    result_cv.to_csv(sub_file_cv, index=True, index_label='id', compression='gzip')
    