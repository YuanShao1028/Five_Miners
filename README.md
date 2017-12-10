# Five_Miners

Project link:<br>
https://github.com/YuanShao1028/Five_Miners

Data source:<br>
KKBOX: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data


File Description:<br>
feature.py: data preprocessing & feature engineering<br>
train.py: GBDT training<br>
meancode_transfer.py: Mean Encoding for feature transformation.<br>
xgb.py: (Follow Up) stacking GBDT model with random forest model.


Run:
1. generate preprocessed train & test dataset
\# set dataset path and output path in feature.py
```
python feature.py  
```

\# [optional] Mean Encoding
```
# python meancode_transfer.py
```

2. model training and testing
```
python train.py 
```
