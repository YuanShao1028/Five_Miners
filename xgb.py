#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:24:44 2017

@author: Yuan Shao
"""
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.externals import joblib
#xgb 0.670
#lgb 0.680
#rf
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
train_df = pd.read_csv('num_train.csv')
test_df = pd.read_csv('num_test.csv')

#rf
x_test = test.drop(['id'], axis=1)
ids = test['id'].values

x = train_df.drop('target',axis = 1)
y = train_df['target'].values

clf_rf = RandomForestClassifier(n_estimators = 40,
                                criterion = 'gini',
                                max_depth = 10,
                                min_samples_split = 4,
                                min_samples_leaf = 1,
                                max_features = 'auto',
                                oob_score = True)
clf_rf.fit(x,y)

result_rf = clf_rf.predict_proba(test_df.drop('id',axis = 1))
result = []
for i in range(len(result_rf)):
    result.append(max(result_rf[0]))
ids = test_df['id'].values




#joblib.dump(clf, "model_rf.m") #存储
#clf_rf = joblib.load("model_rf.m") #调用

subm = pd.DataFrame()
subm['id'] = ids.astype('int32')
subm['target'] = result
subm.to_csv('submission_rf.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')




































































































