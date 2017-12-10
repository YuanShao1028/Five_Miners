#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:43:11 2017

@author: Yuan Shao
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import random
from sklearn import metrics
from collections import Counter
import sklearn
train = pd.read_csv('train_1_eng')
test = pd.read_csv('test_1_eng')


col = list(train.columns)
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
category_col = ['gender',
                'language',
                'lyricist',
                'composer',
                'artist_name',
                'genre_ids',
                'source_type',
                'source_screen_name',
                'source_system_tab',
                'song_id',
                'msno',
                'city',
                'song_length_class']

numerical_col = ['song_length',
                 'bd',#outlier tackle
                 'registered_via',
                 'expiration_date',
                 'membership_days',
                 'registration_year',
                 'registration_month',
                 'registration_date',
                 'expiration_year',
                 'expiration_month',
                 'song_year',#fillna
                 'lyricists_count',
                 'composer_count',
                 #'is_featured',
                 #'artist_count',
                 #'artist_composer',
                 #'artist_composer_lyricist',
                 #'song_lang_boolean',
                 #'smaller_song',
                 'count_song_played',
                 'count_artist_played']





def age_transfer(age):#deal with bd's outlier
    new_age = age
    if age == 0:
        new_age = random.randint(7,50)
    if age <= 7 and age > 0:
        new_age = random.randint(12,25)
    if age >= 75:
        new_age = random.randint(45,75)
    return new_age
train['bd'] = train['bd'].apply(age_transfer).astype(np.int64)
test['bd'] = test['bd'].apply(age_transfer).astype(np.int64)

def song_length_class(length):
    song_class = 'Unknown'
    if length <= 1.8e+05:
        song_class = 'short_song'
    if length >= 1.8e+05 and length <= 3.0e+05:
        song_class = 'normal_song'
    if length >= 3.0e+05:
        song_class = 'long_song'
    return song_class
train['song_length_class'] = train['song_length'].apply(song_length_class).astype('category')
test['song_length_class'] = test['song_length'].apply(song_length_class).astype('category')


train['song_year'] = train['song_year'].fillna(train['song_year'].median()).astype('int64')
test['song_year'] = test['song_year'].fillna(test['song_year'].median()).astype('int64')
train['city'] = train['city'].astype('category')
test['city'] = test['city'].astype('category')
train['language'] = train['language'].astype('category')
test['language'] = test['language'].astype('category')

drop = []
for each in col:
    if (each in list(model.feature_name())) == False:
        drop.append(each)
drop.remove('target')

for each in drop:
    train = train.drop(each,axis = 1)
    test = test.drop(each, axis = 1)

#model_train_area

params_gdbt = {
        'objective': 'binary',#
        'boosting': 'gbdt',#
        'learning_rate': 0.1 ,#
        'verbose': 0,#
        'num_leaves': 108,#
        'bagging_fraction': 0.95,#
        'bagging_freq': 1,#
        'bagging_seed': 1,#
        'feature_fraction': 0.9,#
        'feature_fraction_seed': 1,#
        'max_bin': 512,#
        'max_depth': 16,#
        'num_rounds': 200,
        'metric' : 'auc'
    }

x_all = train.drop('target',axis = 1)
y_all = train['target'].values
d_train_all = lgb.Dataset(x_all,y_all)
watchlist_all = lgb.Dataset(x_all,y_all)
model_gdbt = lgb.train(params_gdbt, train_set=d_train_all,  valid_sets=watchlist_all, verbose_eval=5)








#output area
p_test_gdbt = model.predict(test.drop('id',axis = 1))
subm = pd.DataFrame()
subm['id'] = test['id'].values
subm['target'] = p_test_gdbt
subm.to_csv('submission_0.6875.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

#local validation
#divided validation
train_val = train.tail(2500000)
train_trn = train.head(len(train)- 2500000)
#training
model_new = lgb.LGBMClassifier(boosting_type = 'gbdt',
                               num_leaves = 108,
                               max_depth = -1,
                               learning_rate = 0.1,
                               objective = 'binary',
                               bagging_fraction = 0.95,
                               bagging_freq = 1,
                               #bagging_seed = 1,
                               feature_fraction = 0.9,
                               #feature_fraction_seed = 1
                               num_rounds = 200,
                               metric = 'auc'
                               )
X = train_trn.drop('target',axis = 1)
y = train_trn['target'].values
model_new.fit(X,y)
#predict result
local_result = model_new.predict(train_val.drop('target',axis = 1))
score = metrics.roc_auc_score(train_val['target'], pred_y)



'''@@@@@@@@@@@@@@@@@@@@@@@tool area@@@@@@@@@@@@@@@@@@@@@@@'''
train['song_year'].describe()
Counter(train['artist_name'])
Counter(train['source_type'])
len(x = list(x_test.columns)


for each in train.columns:
    print(str(each) + (25-len(each))*' ' + str(train[each].dtype))
for each in train.columns:
    print(str(each) + (25-len(each))*' ' + str(train[each].isnull().values.any()))
feature_importance_new = pd.Series(index = model.feature_name(),data = model_new.feature_importance())

model = lgb.Booster(model_file = '/Users/apple/Desktop/KKBOX/models/0.6875')




