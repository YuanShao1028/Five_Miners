# Five_Miners

Project link:
https://github.com/YuanShao1028/Five_Miners

Data source:
Dataset from KKBOX: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data


File Description:
feature.py: data preprocessing & feature engineering
train.py: GBDT training
meancode_transfer.py: Mean Encoding for feature transformation.
xgb.py: (Follow Up) stacking GBDT model with random forest model.


Run:
# 1. generate preprocessed train & test dataset
# set dataset path and output path in feature.py
python feature.py  

# [optional] meancode_transfer
# python meancode_transfer.py

# 2. model training and test
python train.py 
