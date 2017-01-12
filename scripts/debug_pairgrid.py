# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 10:41:17 2016

@author: Tony
"""

import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import gc

DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'
IMG_PATH = '../images/'

train = pd.read_csv(DATA_TRAIN_PATH, dtype={'id': np.int32})
train['logloss'] = np.log(train['loss'])
test = pd.read_csv(DATA_TEST_PATH, dtype={'id': np.int32})
ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train.drop(['loss'], axis=1), test)).reset_index(drop=True)
num_feats = train_test.drop(
    ['id','logloss'],axis=1).dtypes[train_test.dtypes != "object"].index
cat_feats = [feat for feat in test.columns if 'cat' in feat]
###Sort cat features in order to plot
sorted_cat_feats = [sorted(sorted(train_test[feat].unique().tolist()), 
                           key=lambda x:len(x)) for feat in cat_feats]



for i, col_i in enumerate(cat_feats[:-1]):
    for j, col_j in enumerate(cat_feats[:-1]):
        if i < j:
            if train[col_i].nunique() * train[col_j].nunique() <= 10:
                train['zip_col'] = zip(train[col_i],train[col_j])
                zip_col_order = sorted(sorted(train['zip_col'].unique(),
                                              key=lambda x: x[1]),
                                       key=lambda x: x[0])
                train['zip_col_num'] = 0
                for k in range(len(zip_col_order)):
                    train.loc[train.zip_col == zip_col_order[k],'zip_col_num'] = k
                g = sns.PairGrid(data=train, 
                                 vars=[num_feats.tolist()+['id','logloss']], 
                                 hue='zip_col_num')
                g.map_diag(sns.histplot)
                g.map_lower(sns.kdeplot)
                g.map_upper(plt.scatter, alpha=.1)
                g.savefig('../images/cont_pairwise_cat_colors/train_cont_feats_cat_{},{}.png'\
                    .format(i,j))
                fig = g.fig
                plt.close(fig)
                plt.close('all')
                gc.collect()