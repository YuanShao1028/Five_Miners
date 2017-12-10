#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:02:37 2017

@author: Yuan Shao
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

data_path = '/Users/apple/Desktop/KKBOX/'
output_train = 'train_1_eng'
output_test = 'test_1_eng'


train_path = os.path.expanduser(data_path + 'train.csv')
test_path = os.path.expanduser(data_path + 'test.csv')
songs_path = os.path.expanduser(data_path + 'songs.csv')
members_path = os.path.expanduser(data_path + 'members.csv')
song_extra_path = os.path.expanduser(data_path + 'song_extra_info.csv')

train = pd.read_csv(train_path, dtype={'target' : np.uint8,})
test = pd.read_csv(test_path)
songs = pd.read_csv(songs_path)
members = pd.read_csv(members_path,dtype={'bd' : np.uint8},parse_dates=['registration_init_time','expiration_date'])
songs_extra = pd.read_csv(song_extra_path)





train = train.merge(songs, on='song_id', how='left')
test = test.merge(songs, on='song_id', how='left')

members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)

members['registration_year'] = members['registration_init_time'].dt.year
members['registration_month'] = members['registration_init_time'].dt.month
members['registration_date'] = members['registration_init_time'].dt.day

members['expiration_year'] = members['expiration_date'].dt.year
members['expiration_month'] = members['expiration_date'].dt.month
members['expiration_date'] = members['expiration_date'].dt.day
members = members.drop(['registration_init_time'], axis=1)

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan

songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on = 'song_id', how = 'left')
train.song_length.fillna(200000,inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')


test = test.merge(songs_extra, on = 'song_id', how = 'left')
test.song_length.fillna(200000,inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')




train['genre_ids'] = train['genre_ids'].astype('object')
test['genre_ids'] = test['genre_ids'].astype('object')
train['genre_ids'].fillna('no_genre_id',inplace=True)
test['genre_ids'].fillna('no_genre_id',inplace=True)
train['genre_ids'] = train['genre_ids'].astype('category')
test['genre_ids'] = test['genre_ids'].astype('category')





def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    return sum(map(x.count, ['|', '/', '\\', ';']))
train['lyricist'] = train['lyricist'].astype('object')
test['lyricist'] = test['lyricist'].astype('object')
train['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricist'].fillna('no_lyricist',inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)
train['lyricist'] = train['lyricist'].astype('category')
test['lyricist'] = test['lyricist'].astype('category')







def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
train['composer'].fillna('no_composer',inplace=True)
test['composer'].fillna('no_composer',inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)




train['artist_name'] = train['artist_name'].astype('category')
test['artist_name'] = test['artist_name'].astype('category')
train['composer'] = train['composer'].astype('category')
test['composer'] = test['composer'].astype('category')
train['lyricist'] = train['lyricist'].astype('category')
test['lyricist'] = test['lyricist'].astype('category')




# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}
def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0

train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)
# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}
def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0

train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)
train['language'] = train['language'].astype('category')
test['language'] = test['language'].astype('category')
train['city'] = train['city'].astype('category')
test['city'] = test['city'].astype('category')

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')


train.to_csv(output_train,index = False)
test.to_csv(output_test,index = False)









'''@@@@@@@@@@@@@@@@@@@@@@@tool area@@@@@@@@@@@@@@@@@@@@@@@'''
'''
for each in train.columns:
    print(str(each) + (25-len(each))*' ' + str(train[each].dtype))


category_col = ['gender',
                'language',
                'lyricist',
                'composer',
                'artist_name',
                'genre_ids',
                'source_type',
                'source_screen_name',
                'source_system_tab',
                'song_id msno',
                'city']

numerical_col = ['song_length',
                 'bd',
                 'registered_via',
                 'expiration_date',
                 'membership_days',
                 'registration_year',
                 'registration_month',
                 'registration_date',
                 'expiration_year',
                 'expiration_month',
                 'song_year',
                 'genre_ids_count',
                 'lyricists_count',
                 'composer_count',
                 'is_featured',
                 'artist_count',
                 'artist_composer',
                 'artist_composer_lyricist',
                 'song_lang_boolean',
                 'smaller_song',
                 'count_song_played',
                 'count_artist_played']
corr = pd.Series(index = numerical_col)
for each in numerical_col:
    c = train[each].corr(train['target'])
    corr[each] = c


'''
















































