# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 22:55:49 2016

@author: Tony
"""

import cPickle
import string
import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import gc
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'
LIN_DATA_TRAIN_PATH = '../input/lin_train.csv'
LIN_DATA_TEST_PATH = '../input/lin_test.csv'
IMG_PATH = '../images/'

### These variables perform better with OHE
ohe_list = ['cat73', 'cat93', 'cat112']

### These dicts are to re-encode cat112 to state abbreviations
swap_to_num = {'cat112': {'A': 0, 'AA': 25, 'AB': 26, 'AC': 27, 'AD': 28,
'AE': 29, 'AF': 30, 'AG': 31, 'AH': 32, 'AI': 33, 'AJ': 34, 'AK': 35, 'AL': 36,
'AM': 37, 'AN': 38, 'AO': 39, 'AP': 40, 'AQ': 41, 'AR': 42, 'AS': 43, 'AT': 44,
'AU': 45, 'AV': 46, 'AW': 47, 'AX': 48, 'AY': 49, 'B': 1, 'BA': 50, 'C': 2,
'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24}}

swap_to_state = {'cat112': {0: 'AL', 1: 'AK', 2: 'AZ', 3: 'AR', 4: 'CA',
5: 'CO', 6: 'CT', 7: 'DE', 8: 'DC', 9: 'FL', 10: 'GA', 11: 'HI', 12: 'ID',
13: 'IL', 14: 'IN', 15: 'IA', 16: 'KS', 17: 'KY', 18: 'LA', 19: 'ME', 20: 'MD',
21: 'MA', 22: 'MI', 23: 'MN', 24: 'MS', 25: 'MO', 26: 'MT', 27: 'NE', 28: 'NV',
29: 'NH', 30: 'NJ', 31: 'NM', 32: 'NY', 33: 'NC', 34: 'ND', 35: 'OH', 36: 'OK',
37: 'OR', 38: 'PA', 39: 'RI', 40: 'SC', 41: 'SD', 42: 'TN', 43: 'TX', 44: 'UT',
45: 'VT', 46: 'VA', 47: 'WA', 48: 'WV', 49: 'WI', 50: 'WY'}}

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

#def encode_nominal(data, col, LOO=False, shift=False, shift_std=.03):
#    return


### Read the data in and prepare
train = pd.read_csv(DATA_TRAIN_PATH, dtype={'id': np.int32}).iloc[:,1:]
train = train.replace(swap_to_num).replace(swap_to_state)

train = pd.get_dummies(train, columns=ohe_list)
convert_cat_to_num(train)
train['logloss'] = np.log(train['loss'])
test = pd.read_csv(DATA_TEST_PATH, dtype={'id': np.int32}).iloc[:,1:]
test = test.replace(swap_to_num).replace(swap_to_state)
test = pd.get_dummies(test, columns=ohe_list)
convert_cat_to_num(test)
ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train.drop(['loss','logloss'], axis=1), test))\
             .reset_index(drop=True)
num_feats = [feat for feat in test.columns if 'cont' in feat]
lin_feats = [feat for feat in test.columns if 'lin' in feat]
for col in lin_feats:
    train[col] = train[col]/float(train_test[col].max())

cat_feats = [feat for feat in test.columns if 'cat' in feat]
prob_ordered=[]
for elem in cat_feats:
    if train_test[elem].nunique()!=train_test[elem].max()+1:
        prob_ordered.append(elem)

                           
##################################
### Visualize distributions ######
##################################
def plot_cat_vs_cont(col, path=None, transform=None, ohe=False, num=False, size=10):
    if path == None:
        path = '../images/tmp.png'.format(col)
    return

def plot_cont_cont(x_col, y_col='logloss', path='', size=10, x_trans=None,
                      y_trans=None, ohe=False, data=train):
    ### Scatter plots a continuous var against another continuous var
    ### transform
    if path == '':
        path = '../images/bivariate_{}_{}.png'.format(x_col, y_col)
    if ohe != False:
        ### To implement later
        pass
    if x_trans == None:
        g = sns.jointplot(x=x_col, y=y_col, data=data, kind='scatter', 
                          size=size, alpha=.07)
    else:
        train['x_tmp'] = x_trans(data[x_col])
        g = sns.jointplot(x='x_tmp', y=y_col, data=data, kind='scatter', 
                          size=size, alpha=.07)
        train.drop('x_tmp', axis=1, inplace=True)
    fig = g.fig
    g.savefig(path)
    plt.close(fig)
    plt.close('all')
    gc.collect()
    return

def plot_resid_cont_loss(x_col, y_col='logloss', path='', size=10, data=train):
    tmp_data = data[[x_col,y_col]].copy()
    linear_model = smf.ols(y_col +' ~ '+ x_col, tmp_data).fit()
    tmp_data['residuals'] = linear_model.resid
    tmp_data[x_col] = pd.cut(tmp_data[x_col], bins=50)
    tmp_data = tmp_data.groupby(by=x_col)['residuals']
    tmp_data_mean = tmp_data.mean()
    tmp_data_count = tmp_data.count()
    tmp_data = tmp_data.quantile([.25,.5,.75]).unstack()
    tmp_data['MEAN'] = tmp_data_mean
    tmp_data['COUNT_SCALED'] = tmp_data_count/float(tmp_data_count.max())*float(tmp_data[.75].max())
    tmp_data['COUNT'] = tmp_data_count
    ax = tmp_data.iloc[:,:-1].plot()
    fig = ax.get_figure()
    if path == '':
        return ax, fig
    else:
        fig.savefig(path)
        plt.close(fig)
        plt.close('all')
        gc.collect()
        return tmp_data

df_cut = []
for col in num_feats[0:14]:
    path = '../images/residuals/linear_{}_logloss.png'.format(col)
    df_cut.append(plot_resid_cont_loss(col, 'logloss', path))
for col in num_feats[0:14]:
    path = '../images/residuals/linear_{}_loss.png'.format(col)
    df_cut.append(plot_resid_cont_loss(col, 'loss', path))

### Univariate plots
sns.mpl.rc("figure", figsize=(30, 10))

# Discrete variables
for i, col in enumerate(cat_feats):
    g = sns.countplot(x=col, data=train_test)
    f = g.get_figure()
    f.savefig('../images/univariate_dist/train_test_{}.png'.format(col))
    plt.close(f)
    plt.close('all')
    gc.collect()

# Continuous variables
for col in num_feats:
    g = sns.distplot(train_test[col])
    fig=g.get_figure()
    fig.savefig('../images/univariate_dist/train_{}.png'.format(col))
    plt.close(fig)
    plt.close('all')
    gc.collect()

## Bivariate plots with logloss
for i, col in enumerate(cat_feats[:-1]):
    j = train[col].nunique()
    g = sns.factorplot(kind='box', y='logloss', x=col, data=train, size=10, 
                       aspect=(3.+3.*j)/10.)
    g.savefig('../images/bivariate_dist_logloss/train_{}.png'.format(col))
    plt.close(g.fig)
    plt.close('all')
    gc.collect()
g = sns.factorplot(kind='box', y='logloss', x='cat116', data=train, size=10, 
                   aspect=34.8)
g.savefig('../images/bivariate_dist_logloss/train_{}.png'.format(col))
plt.close(g.fig)
plt.close('all')
gc.collect()

for col in num_feats:
    g = sns.jointplot(x=col, y='logloss', data=train, kind='scatter', size=10, alpha=.05)
    fig = g.fig
    g.savefig('../images/bivariate_dist_logloss/train_{}_scatter.png'.format(col))
    plt.close(fig)
    plt.close('all')
    gc.collect()
#g = sns.jointplot(x='id', y='logloss', data=train, kind='scatter', size=10, alpha=.05)
#fig = g.fig
#g.savefig('../images/bivariate_dist_logloss/train_{}_kde.png'.format('id'))
#plt.close(fig)
#plt.close('all')
#gc.collect()

for col in num_feats:
    g = sns.jointplot(x=col, y='logloss', data=train, kind='', size=10)
    fig = g.fig
    g.savefig('../images/bivariate_dist_logloss/train_{}_hex.png'.format(col))
    plt.close(fig)
    plt.close('all')
    gc.collect()


#g = sns.jointplot(x='id', y='logloss', data=train, kind='hex', size=10)
#fig = g.fig
#g.savefig('../images/bivariate_dist_logloss/train_{}_hex.png'.format('id'))
#plt.close(fig)
#plt.close('all')
#gc.collect()

for col in num_feats:
    plot_cont_cont(col, 
        path='../images/bivariate_dist_logloss/1train_{}_linear.png'.format(col))
    plot_cont_cont(col, x_trans=lambda x: np.log(x+1e-6),
        path='../images/bivariate_dist_logloss/1train_{}_log.png'.format(col))
    

#g = sns.jointplot(x='id', y='logloss', data=train, kind='hex', size=10)
#fig = g.fig
#g.savefig('../images/bivariate_dist_logloss/train_{}_hex.png'.format('id'))
#plt.close(fig)
#plt.close('all')
#gc.collect()

#sns.mpl.rc("figure", figsize=(1000,1000))
#sns.mpl.rc("axes", titlesize=2)
#sns.mpl.rc("legend", fontsize=2)
for i, col_i in tqdm.tqdm(enumerate(cat_feats[:-1])):
    for j, col_j in tqdm.tqdm(enumerate(cat_feats[:-1])):
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
                                 vars=num_feats.tolist()+['id','logloss'], 
                                 hue='zip_col_num', size=4)
                g.map_diag(sns.kdeplot)
                g.map_offdiag(plt.scatter, alpha=.2)
                g.savefig('../images/cont_pairwise_cat_colors/train_cont_feats_cat_{}_{}.png'\
                    .format(i,j))
                fig = g.fig
                plt.close(fig)
                plt.close('all')
                gc.collect()

### To check some of the outliers at left and right of bivariate graphs
num_stats = pd.DataFrame(np.zeros((len(num_feats),6)), index=num_feats,
                         columns=['count_min','count_min+.03','count_min+.1',\
                         'count_max-.1','count_max-.03','count_max'])
for elem in num_feats:
    col_min = train[elem].min()
    col_max = train[elem].max()
    num_stats.loc[elem,'count_min']=train[elem][train[elem]==col_min].count()
    num_stats.loc[elem,'count_min+.03']=train[elem][train[elem]<=col_min+.03].count()
    num_stats.loc[elem,'count_min+.1']=train[elem][train[elem]<=col_min+.1].count()
    num_stats.loc[elem,'count_max-.1']=train[elem][train[elem]>=col_max-.1].count()
    num_stats.loc[elem,'count_max-.03']=train[elem][train[elem]>=col_max-.03].count()
    num_stats.loc[elem,'count_max']=train[elem][train[elem]==col_max].count()

### *** NEED TO COMPARE UNIVARIATE DISTRIBUTIONS FOR < 90TH PCT AND >= 90TH PCT

train_90 = train[train.loss<train.loss.quantile(.9)].reset_index().drop(['index'], axis=1)
train_10 = train[train.loss>=train.loss.quantile(.9)].reset_index().drop(['index'], axis=1)

######################################
####### Time Series ##################
######################################
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

plt.rcParams['figure.figsize'] = 10, 6

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

def plot_acf_pacf(ts, nlags=0, save_file=""):
    plt.clf()
    plt.rcParams['figure.figsize'] = 15, 10
    #ACF and PACF plots:
    if nlags == 0:
        nlags=min(ts.shape[0]-1,85)
    lag_acf = acf(ts, nlags=nlags)
    lag_pacf = pacf(ts, nlags=nlags, method='ols')

    #Plot ACF: 
    plt.subplot(211) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(212)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    if save_file != "":
        plt.savefig(save_file)


### Plot differences before and after linearization w/ACF & PACF
for elem in num_feats:
    lin_col = lintrain[['lin_' + elem, 'logloss']].astype('int32')
    lin_col = lin_col.groupby('lin_'+elem)
    lin_col = lin_col.aggregate([np.median, np.mean])
    plot_acf_pacf(lin_col.iloc[:,0], save_file=IMG_PATH+'ts/'+'lin_'+elem+'_median.png')
    plot_acf_pacf(lin_col.iloc[:,1], save_file=IMG_PATH+'ts/'+'lin_'+elem+'_mean.png')



def decompose_ts(ts):
    decomposition = seasonal_decompose(ts)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    return trend, seasonal, residual

##################################
### Feature Engineering ##########
##################################

### Check if each feature should be one hot encoded or labeled as a number
import model_eval
folds = 5
early_stop = 25
### Through HyperOpt, found best log_y value to be 
log_y = 436.28279443805883

scores = {}

scores['base_log'] = model_eval.xgb_quick_eval(train.drop(['loss','logloss'], axis=1),
    np.log(train['loss']+log_y), folds=folds, log_y=log_y, booster='gbtree', early_stop=early_stop)
scores['base'] = model_eval.xgb_quick_eval(train.drop(['loss','logloss'], axis=1),
    train['loss'], folds=folds, log_y=False, booster='gbtree', early_stop=early_stop)
for feat in cat_feats:
    if train[feat].nunique() > 2:
        print('Checking feature {}'.format(feat))
        train_ohe = pd.concat([train, pd.get_dummies(train[feat].astype('string'))],
                               axis=1).drop([feat,'logloss','loss'], axis=1)
        scores[feat] = model_eval.xgb_quick_eval(train_ohe, train['logloss'], 
                                              folds=folds, log_y=True)


scores_qr = {}

scores_qr['base'] = model_eval.qr_quick_eval(train.drop(['loss','logloss'], axis=1),
                                        train['loss'], folds=folds)
for feat in cat_feats:
    if train[feat].nunique() > 2:
        print('Checking feature {}'.format(feat))
        train_ohe = pd.concat([train, pd.get_dummies(train[feat].astype('string'))],
                               axis=1).drop([feat,'logloss','loss'], axis=1)
        scores_qr[feat] = model_eval.qr_quick_eval(train_ohe, train['loss'], 
                                              folds=folds)
                                              
scores_lr = {}

scores_lr['base'] = model_eval.lr_quick_eval(train.drop(['loss','logloss'], axis=1),
                                        train['logloss'], folds=folds, log_y=True)
for feat in cat_feats:
    if train[feat].nunique() > 2:
        print('Checking feature {}'.format(feat))
        train_ohe = pd.concat([train, pd.get_dummies(train[feat].astype('string'))],
                               axis=1).drop([feat,'logloss','loss'], axis=1)
        scores_qr[feat] = model_eval.lr_quick_eval(train_ohe, train['loss'], 
                                              folds=folds)