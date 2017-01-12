# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 13:22:08 2016

@author: Tony
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from numba import jit

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import string

shift = 450
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
def fair_obj(preds, dtrain, fair_constant=0.7):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = fair_constant / (abs(x) + fair_constant)
    grad = den * x
    hess = den * den
    return grad, hess

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(exp0(y),exp0(yhat))
                                      
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

    n_folds = 10
    early_stopping = 200
    xgb_rounds = []

    d_test = xgb.DMatrix(test_x)
    del test
    
    cv_preds = np.zeros(train_ids.shape[0])
    y_preds = np.zeros((test_ids.shape[0], n_folds))

    kf = kfold_regression(train_y, n_folds, 64)
    for i, (train_index, cv_index) in enumerate(kf):
        print('\n Fold %d' % (i+1))
        X_train, X_cv = train_x.iloc[train_index], train_x.iloc[cv_index]
        y_train, y_cv = train_y.iloc[train_index], train_y.iloc[cv_index]

        rand_state = 2016

        params = {
            'seed': 0,
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.03,
            'objective': 'reg:linear',
            'max_depth': 12,
            'min_child_weight': 100,
            'base_score': 7.75,
            'booster': 'gbtree'}

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_cv = xgb.DMatrix(X_cv, label=y_cv)
        watchlist = [(d_train, 'train'), (d_cv, 'eval')]

        clf = xgb.train(params,
                        d_train,
                        100000,
                        watchlist,
                        verbose_eval=50,
                        early_stopping_rounds=200,
                        obj=fair_obj,
                        feval=xg_eval_mae)

        xgb_rounds.append(clf.best_iteration)
        cv_preds[cv_index] = exp0(clf.predict(d_cv, ntree_limit=clf.best_ntree_limit))
        cv_score = mean_absolute_error(exp0(y_cv), cv_preds[cv_index])
        print('eval-MAE: %.6f' % cv_score)
        y_preds[:,i] = exp0(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))
    
    score = mean_absolute_error(exp0(train_y), cv_preds)
    print('Average eval-MAE: %.6f' % score)

    print("Writing test results")
    result = pd.DataFrame(y_preds.mean(1), columns=['loss'])
    result["id"] = test_ids
    result = result.set_index("id")
    print("%d-fold average prediction:" % n_folds)

    now = datetime.now()
    score = str(round(score, 5))
    sub_file = '../L0/submission_{}fold-average-xgb_fairobj_{}_{}.csv.gz'\
        .format(n_folds, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='id', compression='gzip')
    
    print("Writing cv results")
    result_cv = pd.DataFrame(cv_preds, columns=['loss'])
    result_cv["id"] = train_ids
    result_cv = result_cv.set_index("id")

    sub_file_cv = '../L0/oof_{}fold-average-xgb_fairobj_{}_{}.csv.gz'\
        .format(n_folds, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("Writing submission: %s" % sub_file_cv)
    result_cv.to_csv(sub_file_cv, index=True, index_label='id', compression='gzip')
    