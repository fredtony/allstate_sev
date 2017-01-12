# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:57:40 2016

@author: Tony
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox, shapiro, gmean
from statsmodels.stats.diagnostic import normal_ad
from math import exp, log
import xgboost as xgb


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))


def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'


def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train_loader = pd.read_csv(path_train, dtype={'id': np.int32})
    train = train_loader.drop(['id', 'loss'], axis=1)
    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
    test = test_loader.drop(['id'], axis=1)
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)
    numeric_feats = train_test.dtypes[train_test.dtypes != "object"].index
    
     ### compute skew and do Box-Cox transformation
    skewed_feats = train_test[numeric_feats].apply(lambda x: abs(skew(x.dropna())))
    print("\nSkew in numeric features before transformations:")
    print(skewed_feats)
    shapiro_feats = train_test[numeric_feats].apply(lambda x: shapiro(x)[0])
    print("\nShapiro test statistics in numeric features before transformations:")
    print(shapiro_feats)
    ### transform features with skew > 0.25 (this can be varied to find optimal value)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index.tolist()
    lam = np.zeros(len(skewed_feats))
    for i, feats in enumerate(skewed_feats):
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam[i] = boxcox(train_test[feats])
        
    skewed_feats = train_test[numeric_feats].apply(lambda x: abs(skew(x.dropna())))
    print("\nSkew in numeric features after transformations:")
    print(skewed_feats)
    shapiro_feats = train_test[numeric_feats].apply(lambda x: shapiro(x)[0])
    print("\nShapiro test statistics in numeric features after transformations:")
    print(shapiro_feats)
    
    features = train.columns
    cats = [feat for feat in features if 'cat' in feat]
    # factorize categorical features
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    x_train = train_test.iloc[:ntrain, :]
    x_test = train_test.iloc[ntrain:, :]
    train_test_scaled, scaler = scale_data(train_test)
    train, _ = scale_data(x_train, scaler)
    test, _ = scale_data(x_test, scaler)
    
    train_labels = np.log(np.array(train_loader['loss']))
    train_ids = train_loader['id'].values.astype(np.int32)
    test_ids = test_loader['id'].values.astype(np.int32)

    return train, train_labels, test, train_ids, test_ids

def xgb_eval(X_train_cv, y, X_test, test_ids, folds=5):
    d_test = xgb.DMatrix(X_test)
    
    # set up KFold that matches xgb.cv number of folds
    val_pred = np.zeros(X_train_cv.shape[0])
    y_pred = np.zeros((X_test.shape[0], folds))
    cv_score = []
    kf = KFold(n_splits=folds)#, shuffle=True, random_state=0)
    for i, (train_index, cv_index) in enumerate(kf.split(X_train_cv)):
        print('\n Fold %d\n' % (i + 1))
        X_train, X_cv = X_train_cv[train_index], X_train_cv[cv_index]
        y_train, y_cv = y[train_index], y[cv_index]

    #######################################
    # Define cross-validation variables
    #######################################

        params = {}
        params['booster'] = 'gbtree'
        params['objective'] = "reg:linear"
        params['eval_metric'] = 'mae'
        params['eta'] = 0.1
        params['gamma'] = 0.5290
        params['min_child_weight'] = 4.2922
        params['colsample_bytree'] = 0.3085
        params['subsample'] = 0.9930
        params['max_depth'] = 7
        params['max_delta_step'] = 0
        params['silent'] = 1
        params['random_state'] = 1001
    
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_cv = xgb.DMatrix(X_cv, label=y_cv)
        watchlist = [(d_train, 'train'), (d_cv, 'eval')]

    ####################################
    #  Build Model
    ####################################

        clf = xgb.train(params, d_train, 100000, watchlist, verbose_eval=50,
                        early_stopping_rounds=early_stopping)

    ####################################
    #  Evaluate Model and Predict
    ####################################

        xgb_rounds.append(clf.best_iteration)
        val_pred[cv_index] = (np.exp(clf.predict(d_cv, ntree_limit=clf.best_ntree_limit)))
        cv_score.append(mean_absolute_error(np.exp(y_cv), val_pred[cv_index]))
        y_pred[:,i] = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))

    ####################################
    #  Add Predictions and Average Them
    ####################################

    mpred = np.average(y_pred, axis=1)
    gpred = gmean(y_pred, axis=1)
    score = np.average(cv_score)
    print('\n CV scores:')
    for i in range(folds):
        print('eval-MAE fold {0}: {1:.6f}'.format(i, cv_score[i]))
    print('\n Average eval-MAE: {0:.6f}'.format(score))
    print('\n Folded eval-MAE: {0:.6f}'.format(mean_absolute_error(np.exp(y), val_pred)))

    #################################################
    ###  Finalize prediction and write submission ###
    #################################################

    timer(start_time)

    print("#\n Writing results")
    result = pd.DataFrame(mpred, columns=['loss'])
    result["id"] = test_ids
    result = result.set_index("id")
    print("\n {}-fold average prediction:\n".format(folds))
    print(result.head(10))
    
    result_geom = pd.DataFrame(gpred, columns=['loss'])
    result_geom["id"] = test_ids
    result_geom = result_geom.set_index("id")
    print("\n {}-fold average prediction:\n".format(folds))
    print(result_geom.head(10))
    #result_full = pd.DataFrame(y_pred_full, columns=['loss'])
    #result_full["id"] = ids
    #result_full = result_full.set_index("id")
    #print("\n Full dataset prediction:\n")
    #print(result_full.head())
    #result_fixed = pd.DataFrame(y_pred_fixed, columns=['loss'])
    #result_fixed["id"] = ids
    #result_fixed = result_fixed.set_index("id")
    #print("\n Full datset (at CV #iterations) prediction:\n")
    #print(result_fixed.head())
    
    now = datetime.now()

    #score = str(round((cv_sum / folds), 6))
    sub_file = 'submission_{0}fold-average-xgb_{1}_{2}.csv.gz'.format(
        folds, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='id', compression='gzip')
    
    sub_file = 'submission_{0}fold-geometric-average-xgb_{1}_{2}.csv.gz'.format(
        folds, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing submission: %s" % sub_file)
    result_geom.to_csv(sub_file, index=True, index_label='id', compression='gzip')
    
    #sub_file = 'submission_full-average-xgb_' + str(now.strftime(
    #    "%Y-%m-%d-%H-%M")) + '.csv.gz'
    #print("\n Writing submission: %s" % sub_file)
    #result_full.to_csv(sub_file, index=True, index_label='id', compression='gzip')
    
    #sub_file = 'submission_full-CV-xgb_' + str(now.strftime(
    #    "%Y-%m-%d-%H-%M")) + '.csv.gz'
    #print("\n Writing submission: %s" % sub_file)
    #result_fixed.to_csv(sub_file, index=True, index_label='id', compression='gzip')
    
################################## Actual Run Code ##################################


# enter the number of folds from xgb.cv
folds = 5
early_stopping = 25
xgb_rounds = []

start_time = timer(None)

# Load data set and target values
X_train_cv, y, X_test, _, test_ids = load_data()
xgb_eval(X_train_cv, y, X_test, test_ids, folds=folds)
