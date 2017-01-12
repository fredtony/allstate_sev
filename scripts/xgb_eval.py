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
#from scipy.stats import skew, boxcox, shapiro
from scipy.stats import gmean
import xgboost as xgb
from statsmodels.api import add_constant
from statsmodels.regression.quantile_regression import QuantReg


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
#    skewed_feats = train_test[numeric_feats].apply(lambda x: abs(skew(x.dropna())))
#    print("\nSkew in numeric features before transformations:")
#    print(skewed_feats)
#    shapiro_feats = train_test[numeric_feats].apply(lambda x: shapiro(x)[0])
#    print("\nShapiro test statistics in numeric features before transformations:")
#    print(shapiro_feats)
#    ### transform features with skew > 0.25 (this can be varied to find optimal value)
#    skewed_feats = skewed_feats[skewed_feats > 0.25]
#    skewed_feats = skewed_feats.index.tolist()
#    lam = np.zeros(len(skewed_feats))
#    for i, feats in enumerate(skewed_feats):
#        train_test[feats] = train_test[feats] + 1
#        train_test[feats], lam[i] = boxcox(train_test[feats])
#        
#    skewed_feats = train_test[numeric_feats].apply(lambda x: abs(skew(x.dropna())))
#    print("\nSkew in numeric features after transformations:")
#    print(skewed_feats)
#    shapiro_feats = train_test[numeric_feats].apply(lambda x: shapiro(x)[0])
#    print("\nShapiro test statistics in numeric features after transformations:")
#    print(shapiro_feats)
    
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

def xgb_eval(X_train_cv, y, X_test, test_ids, folds=5, log_y=True, early_stop=25):
    xgb_rounds = []    
    start_time = timer(None)
    d_test = xgb.DMatrix(X_test)
    
    # set up KFold that matches xgb.cv number of folds
    val_pred = np.zeros(X_train_cv.shape[0])
    y_pred = np.zeros((X_test.shape[0], folds))
    cv_score = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    for i, (train_index, cv_index) in enumerate(kf.split(X_train_cv)):
        print('\n Fold {}'.format(i+1))
        X_train, X_cv = X_train_cv.iloc[train_index,:], X_train_cv.iloc[cv_index,:]
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

        clf = xgb.train(params, d_train, 100000, watchlist,
                        early_stopping_rounds=early_stop)

    ####################################
    #  Evaluate Model and Predict
    ####################################
        
        xgb_rounds.append(clf.best_iteration)
        if log_y == True:
            val_pred[cv_index] = (np.exp(clf.predict(d_cv, ntree_limit=clf.best_ntree_limit)))
            cv_score.append(mean_absolute_error(np.exp(y_cv), val_pred[cv_index]))
            y_pred[:,i] = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))
        else:
            val_pred[cv_index] = clf.predict(d_cv, ntree_limit=clf.best_ntree_limit)
            cv_score.append(mean_absolute_error(y_cv, val_pred[cv_index]))
            y_pred[:,i] = clf.predict(d_test, ntree_limit=clf.best_ntree_limit)
        

    ####################################
    #  Add Predictions and Average Them
    ####################################

    mpred = np.average(y_pred, axis=1)
    gpred = gmean(y_pred, axis=1)
    score = np.average(cv_score)
    gscore = gmean(cv_score)
    print('\n CV scores:')
    for i in range(folds):
        print('eval-MAE fold {0}: {1:.6f}'.format(i, cv_score[i]))
    print('\n Average eval-MAE: {0:.6f}'.format(score))
    print('\n Geom Average eval-MAE: {0:.6f}'.format(gscore))
    if log_y:
        print('\n Folded eval-MAE: {0:.6f}'.format(mean_absolute_error(np.exp(y), val_pred)))
    else:
        print('\n Folded eval-MAE: {0:.6f}'.format(mean_absolute_error(y, val_pred)))

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


def xgb_quick_eval(X_train_cv, y, folds=5, log_y=True, early_stop=25):
    start_time = timer(None)
    
    # set up KFold that matches xgb.cv number of folds
    val_pred = np.zeros(X_train_cv.shape[0])
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    train_index, cv_index = next(iter(kf.split(X_train_cv)))
    X_train, X_cv = X_train_cv.iloc[train_index,:], X_train_cv.iloc[cv_index,:]
    y_train, y_cv = y[train_index], y[cv_index]

    #######################################
    # Define cross-validation variables
    #######################################

    params = {}
    params['booster'] = 'gbtree'
    params['objective'] = "reg:linear"
    params['eval_metric'] = 'mae'
    params['eta'] = 0.15
    params['gamma'] = 0.5290
    params['min_child_weight'] = 4.2922
    params['colsample_bytree'] = 0.3085
    params['subsample'] = 0.9930
    params['max_depth'] = 7
    params['max_delta_step'] = 0
    params['silent'] = 1
    params['random_state'] = 1#1001

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_cv = xgb.DMatrix(X_cv, label=y_cv)
    watchlist = [(d_train, 'train'), (d_cv, 'eval')]

    ####################################
    #  Build Model
    ####################################

    clf = xgb.train(params, d_train, 100000, watchlist,
                    early_stopping_rounds=early_stop, verbose_eval=50)

    ####################################
    #  Evaluate Model and Predict
    ####################################
    
    if log_y:
        val_pred = np.exp(clf.predict(d_cv, ntree_limit=clf.best_ntree_limit))
        score = mean_absolute_error(np.exp(y_cv), val_pred)
    else:
        val_pred = clf.predict(d_cv, ntree_limit=clf.best_ntree_limit)
        score = mean_absolute_error(y_cv, val_pred)
        

    ####################################
    #  Add Predictions and Average Them
    ####################################

    print('Average eval-MAE: {0:.6f}'.format(score))
    if log_y:
        print ('\nMeans: actual={}, predicted={}'.format(np.mean(np.exp(y)), 
                                                       np.mean(val_pred)))
        print ('Medians: actual={}, predicted={}'.format(np.median(np.exp(y)), 
                                                       np.median(val_pred)))
    else:
        print ('\nMeans: actual={}, predicted={}'.format(np.mean(y), 
                                                       np.mean(val_pred)))
        print ('Medians: actual={}, predicted={}'.format(np.median(y), 
                                                       np.median(val_pred)))
    timer(start_time)
    return score

def gradient_descent_MAE(X, y, T, alpha):
    m,n = X.shape
    theta = np.zeros(n)
    f = np.zeros(T)
    for i in range(T):
        f[i] = np.linalg.norm(X.dot(theta) - y,1)
        g = X.T.dot(np.sign(X.dot(theta) - y))
        theta = theta - alpha*g
    return theta, f

def qr_quick_eval(X_tr, y, folds=5):
    start_time = timer(None)
    
    X_train_cv, scaler = scale_data(X_tr)
    X_train_cv = add_constant(X_train_cv)
    
    cv_pred = np.zeros(X_train_cv.shape[0])
    cv_score = np.zeros(folds)
    theta_gd = [[] for i in range(folds)]
    maes = [[] for i in range(folds)]
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    for i, (train_index, cv_index) in enumerate(kf.split(X_train_cv)):
        X_train, X_cv = X_train_cv[train_index], X_train_cv[cv_index]
        y_train, y_cv = y[train_index], y[cv_index]
    
        theta_gd[i], f_gd = gradient_descent_MAE(X_train, y_train, 500, .002)
        maes[i] = f_gd/float(X_train.shape[0])
        cv_score[i] = (maes[i][-1])
        print('Fold {} MAE: {}'.format(i+1, cv_score[i]))
        cv_pred[cv_index] = abs(X_cv.dot(theta_gd[i])-y_cv)
    
    mae = np.mean(cv_pred)    
    timer(start_time)
    print('Avg fold score:{}, Overall score:{}'.format(np.mean(cv_score), mae))
    return mae, maes


################################## Actual Run Code ##################################

if __name__ == '__main__':
    # enter the number of folds from xgb.cv
    folds = 5
    early_stopping = 25
    
    # Load data set and target values
    X_train_cv, y, X_test, _, test_ids = load_data()
    xgb_eval(X_train_cv, y, X_test, test_ids, folds=folds)