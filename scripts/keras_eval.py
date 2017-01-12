# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 01:32:03 2016

@author: Tony
"""

import numpy as np

import pandas as pd
import datetime
import subprocess
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
import cPickle
import string
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import gc

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: {} min and {} sec'.format(tmin, round(tsec, 2)))

## read data
DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'
IMG_PATH = '../images/'


### Helper functions
def convert_cat_to_num(data):
    translation = list(string.ascii_uppercase)[:-1]
    for elem_i in translation[:13]:
        for elem_j in translation[:25]:
            translation.append(elem_i + elem_j)
    for i in range(len(translation)):
        data.replace(translation[i], i, inplace=True)
    data.replace('ZZ', -1, inplace=True)
    return
    
### Read the data in and prepare
train = pd.read_csv(DATA_TRAIN_PATH, dtype={'id': np.int32})
convert_cat_to_num(train)
train['logloss'] = np.log(train['loss'])
test = pd.read_csv(DATA_TEST_PATH, dtype={'id': np.int32})
convert_cat_to_num(test)
ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train.drop(['loss'], axis=1), test)).reset_index(drop=True)
num_feats = [feat for feat in test.columns if 'cont' in feat]
cat_feats = [feat for feat in test.columns if 'cat' in feat]
prob_ordered=[]
for elem in cat_feats:
    if train_test[elem].nunique()!=train_test[elem].max()+1:
        prob_ordered.append(elem)

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

###################################################################################

## neural net
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
    return(model)

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