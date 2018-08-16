# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:57:55 2018

@author: ukalwa
"""

import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
# import ast
import os
import json

#import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report, make_scorer, precision_score
from imblearn.over_sampling import SMOTE

cols = ['A1','A2','A_B', 'A_BG', 'A_DB', 'A_LB', 'A_W','B', 'C',
        'D1', 'D2', 'result']
print("PH2 + Mednode test")
#paths = [r'Z:\Upender\complete_mednode_dataset']
paths = [r'Z:\Upender\result_set\400_iterations',
         'Z:\Upender\complete_mednode_dataset']
benign_df = pd.DataFrame()
melanoma_df = pd.DataFrame()
for PATH in paths:
    benign_stats_file = os.path.join(PATH,"benign","results.json")
    melanoma_stats_file = os.path.join(PATH,"melanoma","results.json")


    # Add class 0 for benign
    lines = [json.loads(line.strip())+[0] for line in open(benign_stats_file,
             'r')]
    benign_headers = lines.pop(0)[:-1]+[u"result"]

    df = pd.DataFrame(lines, columns=benign_headers)
    benign_df = pd.concat([df, benign_df],axis=0).reset_index()
    benign_df = benign_df.loc[:,benign_headers]

    # Add class 1 for melanoma
    lines = [json.loads(line.strip())+[1] for line in open(melanoma_stats_file,
             'r')]
    melanoma_headers = lines.pop(0)[:-1]+[u"result"]
    df = pd.DataFrame(lines, columns=melanoma_headers)
    melanoma_df = pd.concat([df, melanoma_df],axis=0).reset_index()
    melanoma_df = melanoma_df.loc[:,melanoma_headers]

total_df = pd.concat([benign_df, melanoma_df], axis=0).reset_index()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
sss.get_n_splits(total_df.drop('result',axis=1), total_df['result'])

for train_index, test_index in sss.split(
        total_df.drop('result',axis=1), total_df['result']):
    X_train, X_test = (total_df.loc[train_index,cols[-3:-1]].values,
                       total_df.loc[test_index,cols[-3:-1]].values)
    if len(X_train.shape)==1:
        X_train = np.reshape(X_train, (-1, 1))
        X_test = np.reshape(X_test, (-1, 1))
    y_train, y_test = (total_df.loc[train_index,cols[-1]].values,
                       total_df.loc[test_index,cols[-1]].values)
    X_train_data,X_test_data = (total_df.loc[train_index,'image'],
                                total_df.loc[test_index,'image'])

#X_data = total_df.loc[:, cols[:-1]].as_matrix()
#y = np.uint8(total_df.loc[:, cols[-1]].as_matrix())
#
#X_train, X_test, y_train, y_test = train_test_split(
#                    X_data, y, test_size=0.3, random_state=42,stratify=y,
#                    shuffle=True)

# Transform data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-3, 10, 20)
gamma_range = np.logspace(-3, 5, 20)
#param_grid = dict(gamma=gamma_range, C=C_range, kernel=['poly'])
#param_grid = dict(C=C_range, kernel=['linear'])
param_grid = dict(gamma=gamma_range, C=C_range, kernel=['rbf'])
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
#scorer = make_scorer(precision_score(labels=np.unique(y_train)))
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,
                    scoring='precision')

# Resample data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)
grid.fit(X_res, y_res)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# Test
X_test = scaler.transform(X_test)
sm = SMOTE(random_state=42)
X_test, y_test = sm.fit_sample(X_test, y_test)
TP=0; FP=0; TN=0; FN=0
for i in xrange(len(X_test)):
    if y_test[i] == 1:
        if grid.best_estimator_.predict(X_test[i].reshape(1,2)) == y_test[i]:
            TP += 1
        else:
            FN += 1
    else:
        if grid.best_estimator_.predict(X_test[i].reshape(1,2)) == y_test[i]:
            TN += 1
        else:
            FP += 1

TPR = round(float(TP) / (TP + FN),4)
FPR = round(float(FP) / (FP + TN),4)
Accuracy = round(float(TP + TN)/(TP + TN + FP + FN),4)
if TP + FP != 0:
    PPV = round(float(TP)/ (TP + FP),4)
else:
    PPV = 0
if TN + FN != 0:
    NPV = round(float(TN)/ (TN + FN),4)
else:
    NPV = 0

print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
print("Sensitivity: %s, Specificity: %s, Accuracy: %s, PPV: %s, NPV: %s"
      %(TPR*100, (1-FPR)*100,Accuracy*100,PPV*100, NPV*100))
#result = str(classification_report(
#        grid.best_estimator_.predict(X_test), y_test))



