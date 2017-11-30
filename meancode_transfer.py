#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:45:40 2017

@author: Yuan Shao
"""

'''MeanEncoder'''
from Mean import MeanEncoder
import pandas as pd
import numpy as np
import random
from collections import Counter




train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Meanencoder
train["msno"] = train['msno'].astype('object')
test["msno"] = test['msno'].astype('object')
test["msno"] = test["msno"].fillna(test["msno"].mode()[0])
train["msno"] = train["msno"].fillna(train["msno"].mode()[0])
msno_train = pd.DataFrame(train.msno, columns=['msno'])
msno_test = pd.DataFrame(test.msno, columns=['msno'])
meanencoder = MeanEncoder(categorical_features= ['msno'])
meanfeature_msno_train = meanencoder.fit_transform(msno_train, train.target)
meanfeature_msno_test = meanencoder.transform(msno_test)
train['msno'] = meanfeature_msno_train['msno_pred_0']
test['msno'] = meanfeature_msno_test['msno_pred_0']

train["artist_name"] = train['artist_name'].astype('object')
test["artist_name"] = test['artist_name'].astype('object')
test["artist_name"] = test["artist_name"].fillna(test["artist_name"].mode()[0])
train["artist_name"] = train["artist_name"].fillna(train["artist_name"].mode()[0])
artist_name_train = pd.DataFrame(train.artist_name, columns=['artist_name'])
artist_name_test = pd.DataFrame(test.artist_name, columns=['artist_name'])
meanencoder = MeanEncoder(categorical_features= ['artist_name'])
meanfeature_artist_name_train = meanencoder.fit_transform(artist_name_train, train.target)
meanfeature_artist_name_test = meanencoder.transform(artist_name_test)
train['artist_name'] = meanfeature_artist_name_train['artist_name_pred_0']
test['artist_name'] = meanfeature_artist_name_test['artist_name_pred_0']

train["song_id"] = train['song_id'].astype('object')
test["song_id"] = test['song_id'].astype('object')
test["song_id"] = test["song_id"].fillna(test["song_id"].mode()[0])
train["song_id"] = train["song_id"].fillna(train["song_id"].mode()[0])
song_id_train = pd.DataFrame(train.song_id, columns=['song_id'])
song_id_test = pd.DataFrame(test.song_id, columns=['song_id'])
meanencoder = MeanEncoder(categorical_features= ['song_id'])
meanfeature_song_id_train = meanencoder.fit_transform(song_id_train, train.target)
meanfeature_song_id_test = meanencoder.transform(song_id_test)
train['song_id'] = meanfeature_song_id_train['song_id_pred_0']
test['song_id'] = meanfeature_song_id_test['song_id_pred_0']

train["lyricist"] = train['lyricist'].astype('object')
test["lyricist"] = test['lyricist'].astype('object')
test["lyricist"] = test["lyricist"].fillna(test["lyricist"].mode()[0])
train["lyricist"] = train["lyricist"].fillna(train["lyricist"].mode()[0])
lyricist_train = pd.DataFrame(train.lyricist, columns=['lyricist'])
lyricist_test = pd.DataFrame(test.lyricist, columns=['lyricist'])
meanencoder = MeanEncoder(categorical_features= ['lyricist'])
meanfeature_lyricist_train = meanencoder.fit_transform(lyricist_train, train.target)
meanfeature_lyricist_test = meanencoder.transform(lyricist_test)
train['lyricist'] = meanfeature_lyricist_train['lyricist_pred_0']
test['lyricist'] = meanfeature_lyricist_test['lyricist_pred_0']

train["composer"] = train['composer'].astype('object')
test["composer"] = test['composer'].astype('object')
test["composer"] = test["composer"].fillna(test["composer"].mode()[0])
train["composer"] = train["composer"].fillna(train["composer"].mode()[0])
composer_train = pd.DataFrame(train.composer, columns=['composer'])
composer_test = pd.DataFrame(test.composer, columns=['composer'])
meanencoder = MeanEncoder(categorical_features= ['composer'])
meanfeature_composer_train = meanencoder.fit_transform(composer_train, train.target)
meanfeature_composer_test = meanencoder.transform(composer_test)
train['composer'] = meanfeature_composer_train['composer_pred_0']
test['composer'] = meanfeature_composer_test['composer_pred_0']



#other categorical features
object_col = ['source_system_tab','source_screen_name','language','gender','source_type']
#fill in nan
train['source_system_tab'] = train['source_system_tab'].fillna('Unknown')
test['source_system_tab'] = test['source_system_tab'].fillna('Unknown')
train['source_screen_name'] = train['source_screen_name'].fillna('Unknown')
test['source_screen_name'] = test['source_screen_name'].fillna('Unknown')
train['language'] = train['language'].fillna(train['language'].mode()[0])
test['language'] = test['language'].fillna(test['language'].mode()[0])
train['gender'] = train['gender'].fillna('Unknown')
test['gender'] = test['gender'].fillna('Unknown')


#factorization
df = pd.concat([train_mean, test_mean])
for each in object_col:
    for i in range(len(Counter(train_mean[each]))):
        df[str(each) + str(i)] = df[each].factorize()[0]
    df = df.drop(each,axis = 1)
train_df = df[0 : len(train_mean)]
test_df = df[len(train_mean): ]
train_df = train_df.drop('genre_ids',axis = 1)
test_df = test_df.drop('genre_ids',axis = 1)
train_df = train_df.drop('id',axis = 1)
test_df = test_df.drop('target',axis = 1)
#save the data
train_df.to_csv('num_train.csv',index = False)
test_df.to_csv('num_test.csv',index = False)


#tool area
for each in train_mean.columns:
    print(str(each) + (25-len(each))*' ' + str(train_mean[each].dtype))
for each in train_df.columns:
    print(str(each) + (25-len(each))*' ' + str(train_df[each].dtype))
for each in train_mean.columns:
    print(str(each) + (25-len(each))*' ' + str(train_mean[each].isnull().values.any()))

col = list(train_mean.columns)



