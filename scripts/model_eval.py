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
from scipy.stats import gmean
import xgboost as xgb
from joblib import Parallel, delayed
import multiprocessing

# what are your inputs, and what operation do you want to
# perform on each input. For example...
#inputs = xrange(10)
#def processInput(i):
#    return i * i
#num_cores = multiprocessing.cpu_count()
#results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: {} min and {} sec'.format(tmin, round(tsec, 2)))


def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

fair_constant = 1.548
fair_constant_sq = fair_constant**2
def fair_obj(preds, dtrain):
    x = np.array(preds - dtrain.get_label())
    den = np.absolute(x) + fair_constant
    grad = np.divide(np.multiply(fair_constant, x), den)
    hess = np.divide(fair_constant_sq, np.square(den))
    return grad, hess

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
        params['lambda'] = 1.0
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


def xgb_quick_eval(X_train_cv, y, folds=5, log_y=200, early_stop=25, booster='gbtree'):
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
    if booster == 'gbtree':
        params = {}
        params['booster'] = booster
        params['base_score'] = 2100
#        params['objective'] = fair_obj
        params['eval_metric'] = 'mae'
        params['eta'] = 0.5
        params['gamma'] = 0.5290
        params['min_child_weight'] = 4.2922
        params['colsample_bytree'] = 0.3085
        params['subsample'] = 0.9930
        params['lambda'] = 1.0
        params['max_depth'] = 7
        params['max_delta_step'] = 0
        params['silent'] = 1
        params['random_state'] = 1#1001
    elif booster == 'gblinear':
        params = {}
        params['booster'] = booster
#        params['objective'] = fair_obj
        params['eval_metric'] = 'mae'
        params['lambda'] = 1
        params['alpha'] = 0
        params['silent'] = 1
        params['random_state'] = 1#1001
    elif booster == 'dart':
        params = {}
        params['booster'] = booster
#        params['objective'] = fair_obj
        params['eval_metric'] = 'mae'
        params['eta'] = 0.5
        params['gamma'] = 0.5290
        params['min_child_weight'] = 4.2922
        params['colsample_bytree'] = 0.3085
        params['subsample'] = 0.9930
        params['max_depth'] = 7
        params['max_delta_step'] = 0
        params['sample_type'] = 'uniform'
        params['normalize_type'] = 'tree'
        params['rate_drop'] = 0.3
        params['skip_drop'] = 0.4
        params['silent'] = 1
        params['random_state'] = 1#1001
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_cv = xgb.DMatrix(X_cv, label=y_cv)
    watchlist = [(d_train, 'train'), (d_cv, 'eval')]

    ####################################
    #  Build Model
    ####################################
    if log_y != "":
        params['objective'] = 'reg:linear'
        params['eta'] = .1
        params['base_score'] = 0.5
        clf = xgb.train(params, d_train, 100000, watchlist,
            early_stopping_rounds=early_stop, verbose_eval=200)
    else:
        clf = xgb.train(params, d_train, 100000, watchlist, obj=fair_obj,
                    early_stopping_rounds=early_stop, verbose_eval=200)

    ####################################
    #  Evaluate Model and Predict
    ####################################
    
    if log_y != "":
        if booster == 'gblinear':
            val_pred = np.exp(clf.predict(d_cv))-log_y
        else:
            val_pred = np.exp(clf.predict(d_cv, ntree_limit=clf.best_ntree_limit))-log_y
        score = mean_absolute_error(np.exp(y_cv)-log_y, val_pred)
    else:
        if booster == 'gblinear':
            val_pred = clf.predict(d_cv)
        else:
            val_pred = clf.predict(d_cv, ntree_limit=clf.best_ntree_limit)
        score = mean_absolute_error(y_cv, val_pred)
        

    ####################################
    #  Add Predictions and Average Them
    ####################################

    print('Average eval-MAE: {0:.6f}'.format(score))
    if log_y != "":
        print ('\nMeans: actual={}, predicted={}'.format(np.mean(np.exp(y)-log_y), 
                                                       np.mean(val_pred)))
        print ('Medians: actual={}, predicted={}'.format(np.median(np.exp(y)-log_y), 
                                                       np.median(val_pred)))
    else:
        print ('\nMeans: actual={}, predicted={}'.format(np.mean(y), 
                                                       np.mean(val_pred)))
        print ('Medians: actual={}, predicted={}'.format(np.median(y), 
                                                       np.median(val_pred)))
    timer(start_time)
    return score, clf

def gradient_descent_MAE(X, y, T, alpha, L1=0.0, L2=0.0):
    m,n = X.shape
    theta = np.zeros(n)
    f = np.zeros(T+1)
    for i in range(T):
        f[i] = np.linalg.norm(X.dot(theta) - y,1)
        g = X.T.dot(np.sign(X.dot(theta) - y))
        theta = theta - alpha*g
    f[-1] = np.linalg.norm(X.dot(theta) - y,1)
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
    return mae

def lr_quick_eval(X_tr, y, folds=5, log_y=True):
    start_time = timer(None)
    
    cv_pred = np.zeros(X_tr.shape[0])
    cv_score = np.zeros(folds)
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    for i, (train_index, cv_index) in enumerate(kf.split(X_tr)):
        X_train, X_cv = X_tr.iloc[train_index,:], X_tr.iloc[cv_index,:]
        y_train, y_cv = y[train_index], y[cv_index]
        
        clf = LinearRegression(fit_intercept=True,normalize=True)
        clf.fit(X_train, y_train)
        cv_pred[cv_index] = clf.predict(X_cv)
        cv_score[i] = mean_absolute_error(np.exp(y_cv), np.exp(cv_pred[cv_index]))
        print('Fold {} MAE: {}'.format(i+1, np.exp(cv_score[i])))
    
    mae = mean_absolute_error(np.exp(y), np.exp(cv_pred))
    timer(start_time)
    print(' Avg fold score:{}, Overall score:{}'.format(cv_score.mean(), mae))
    return mae

## Batch generators ######################################################

def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

##############################################################################
###################### Keras Neural Net ######################################
##############################################################################
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint

def create_model(n_input, nodes=[50], reg=1.0, dropouts=[.5], acts=['relu']):
    n_in = n_input    
    model = Sequential()
    for i in xrange(len(nodes)):
        n_out = nodes[i]
        dropout = dropouts[i]
        act = acts[i]
        model.add(Dense(output_dim=n_out, input_dim=n_in, init='he_normal'))
        model.add(act)
        model.add(Dropout(dropout))
        n_in = n_out
    model.add(Dense(output_dim=1, init='he_normal'))
    # Compile model
    adadelta = Adadelta(lr=10.0, rho=0.95, epsilon=1e-08)
    sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mae', optimizer=adadelta)
    return model
    
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = train.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return model

def keras_eval(X_train_cv, y, X_test, test_ids, folds=5, nbags=5, nepochs=55):
    np.random.seed(123)    
    start_time = timer(None)
    
    # set up KFold that matches xgb.cv number of folds
    cv_pred = np.zeros((X_train_cv.shape[0], folds, nbags))
    test_pred = np.zeros((X_test.shape[0], folds, nbags))
    cv_score = np.zeros((folds, nbags))
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    for i, (train_index, cv_index) in enumerate(kf.split(X_train_cv)):
        X_train, X_cv = X_train_cv.iloc[train_index,:], X_train_cv.iloc[cv_index,:]
        y_train, y_cv = y[train_index], y[cv_index]

        ## train models
        for j in range(nbags):
            model = nn_model()
            model.fit_generator(
                generator = batch_generator(X_train, y_train, 128, True),
                nb_epoch = nepochs,
                samples_per_epoch = X_train.shape[0],
                verbose = 0)
            cv_pred[cv_index,i,j] = model.predict_generator(
                generator = batch_generatorp(X_cv, 800, False),
                val_samples = X_cv.shape[0])[:,0]
            test_pred[:,i,j] = model.predict_generator(
                generator = batch_generatorp(X_test, 800, False),
                val_samples = X_test.shape[0])[:,0]
            cv_score[i,j] = mean_absolute_error(y_cv, cv_pred[cv_index])
            print('Fold {}, Bag {} - MAE: {}'.format(i, j, cv_score[i,j]))
        print(' Fold {} - MAE: {}\n'.format(i, cv_score.mean(1)[i]))
    score = mean_absolute_error(y, cv_pred.mean(2).mean(1))
    print('Total - MAE: {}'.format(score))
    timer(start_time)
    
    print("#\n Writing results")
    result = pd.DataFrame({'id': test_ids, 'loss': test_pred.mean(2).mean(1)})
    result = result.set_index("id")
    
    now = datetime.now()

    sub_file = 'submission_{}fold-{}bag-average-keras-{}-{}.csv.gz'.format(
        folds, nbags, score, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing submission: {}".format(sub_file))
    result.to_csv(sub_file, index=True, index_label='id', compression='gzip')

################################## Actual Run Code ##################################

if __name__ == '__main__':
    # enter the number of folds from xgb.cv
    folds = 5
    early_stopping = 25
    
    # Load data set and target values
    X_train_cv, y, X_test, _, test_ids = load_data()
    xgb_eval(X_train_cv, y, X_test, test_ids, folds=folds)