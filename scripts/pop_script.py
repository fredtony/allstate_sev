# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:06:32 2016

@author: Tony
"""

import pandas as pd
import numpy as np
import string
import sys
from scipy.stats import chisquare
import matplotlib as mpl
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'
#POP_PATH = '../input/state_populations.csv'
POPULATION_DATA = StringIO('state_full,state,population\nAlabama,AL,'\
+'4779736\nAlaska,AK,710231\nArizona,AZ,6392017\nArkansas,AR,'\
+'2915918\nCalifornia,CA,37253956\nColorado,CO,5029196\nConnecticut,CT,'\
+'3574097\nDelaware,DE,897934\nDistrict of Columbia,DC,601723\nFlorida,FL,'\
+'18801310\nGeorgia,GA,9687653\nHawaii,HI,1360301\nIdaho,ID,'\
+'1567582\nIllinois,IL,12830632\nIndiana,IN,6483802\nIowa,IA,'\
+'3046355\nKansas,KS,2853118\nKentucky,KY,4339367\nLouisiana,LA,'\
+'4533372\nMaine,ME,1328361\nMaryland,MD,5773552\nMassachusetts,MA,'\
+'6547629\nMichigan,MI,9883640\nMinnesota,MN,5303925\nMississippi,MS,'\
+'2967297\nMissouri,MO,5988927\nMontana,MT,989415\nNebraska,NE,'\
+'1826341\nNevada,NV,2700551\nNew Hampshire,NH,1316470\nNew Jersey,NJ,'\
+'8791894\nNew Mexico,NM,2059179\nNew York,NY,19378102\nNorth Carolina,NC'\
+',9535483\nNorth Dakota,ND,672591\nOhio,OH,11536504\nOklahoma,OK,'\
+'3751351\nOregon,OR,3831074\nPennsylvania,PA,12702379\nRhode Island,RI,'\
+'1052567\nSouth Carolina,SC,4625364\nSouth Dakota,SD,814180\nTennessee,TN,'\
+'6346105\nTexas,TX,25145561\nUtah,UT,2763885\nVermont,VT,625741\nVirginia,VA,'\
+'8001024\nWashington,WA,6724540\nWest Virginia,WV,1852994\nWisconsin,WI,'\
+'5686986\nWyoming,WY,563626\n')

train = pd.read_csv(DATA_TRAIN_PATH, dtype={'id': np.int32})
train['logloss'] = np.log(train['loss'])
test = pd.read_csv(DATA_TEST_PATH, dtype={'id': np.int32})
pop = pd.read_csv(POPULATION_DATA)

translation = list(string.ascii_uppercase)[:-1]
for elem_i in translation[:2]:
    for elem_j in translation[:25]:
        translation.append(elem_i + elem_j)
swap_dict_to_num = {'cat112': dict(zip(translation[:51], np.arange(51)))}
swap_dict_to_state = {'cat112': dict(zip(np.arange(51), pop.state))}

train_test = pd.concat((train.drop(['loss','logloss'], axis=1), test))\
               .reset_index(drop=True).replace(swap_dict_to_num)\
               .replace(swap_dict_to_state)

pop_train_test = train_test.groupby(by='cat112')
counts = pd.DataFrame(pop_train_test.count()['id']).reset_index()
counts.columns = ['state', 'counts']
pop = pop.merge(counts, how='left', on='state')
pop.index = pop.state
pop.drop(['state'], axis=1, inplace=True)

### Calculate expected values based on Census Data 2015
pop['expected'] = pop.population / float(pop.population.sum()) * pop.counts.sum()

### Plot comparison
mpl.rc("figure", figsize=(13,10))
pop[['counts','expected']].plot.bar()

### Perform permutation test on whether randomly shuffled population data
### can get a better chi-squared score than order of data
### P-value represents percentage of times permuted data gets higher chi^2 score
### THIS TEST SORTED BY STATE NAME
sorted_expected = pop.expected.values
sorted_counts = pop.counts.values
chi2_sorted, _ = chisquare(sorted_counts, sorted_expected)

np.random.seed = 0
count = 0
n_tests = 100000
chi2_vals = np.zeros(n_tests)
for i in range(n_tests):
    shuffled_counts = np.random.permutation(sorted_counts)
    chi2_vals[i], _ = chisquare(f_obs=shuffled_counts, f_exp=sorted_counts)
    if chi2_vals[i] < chi2_sorted:
        count += 1
p_val = count/float(n_tests)
print('P-val: {}'.format(p_val))
print('Chi-squared val for ordered by state name: {}'.format(chi2_sorted))


### Now perform same test as if ordered by state abbreviation
abbr_order = np.argsort(pop.index)
pop_2 = pd.DataFrame(pop.values, index=pop.index.values[abbr_order], columns=pop.columns)
pop_2.state_full = pop_2.state_full.values[abbr_order]
pop_2.population = pop_2.population.values[abbr_order]
pop_2.expected = pop_2.expected.values[abbr_order]
pop_2[['counts','expected']].plot.bar()

sorted_expected_2 = pop_2.expected.values
sorted_counts_2 = pop_2.counts.values
chi2_sorted_2, _ = chisquare(sorted_counts_2, sorted_expected_2)

count = 0
chi2_vals_2 = np.zeros(n_tests)
for i in range(n_tests):
    shuffled_counts_2 = np.random.permutation(sorted_counts_2)
    chi2_vals_2[i], _ = chisquare(f_obs=shuffled_counts_2, f_exp=sorted_counts_2)
    if chi2_vals_2[i] < chi2_sorted_2:
        count += 1
p_val = count/float(n_tests)
print('P-val for 2nd test: {}'.format(p_val))
print('Chi-squared val for ordered by state abbr: {}'.format(chi2_sorted_2))