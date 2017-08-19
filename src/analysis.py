# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:34:52 2017

@author: ukalwa
"""

import pandas as pd
import numpy as np
#import ast
import os
import json
#from sklearn import svm
from sklearn import neighbors

path = r'G:\Upender\Melanoma Project\PH2Dataset\Python_script'
benign_stats_file = os.path.join(path,"benign","results.json")
melanoma_stats_file = os.path.join(path,"melanoma","results.json")

#benign_file = os.path.join(path,'benign_nevus.txt')
atypical_file = os.path.join(path,'atypical_nevus.txt')
#melanoma_file = os.path.join(path,'melanoma.txt')

# Load image names to lists to add different class for each
#benign = [line.strip() for line in open(benign_file, 'r')]
atypical = [line.strip() for line in open(atypical_file, 'r')]
#melanoma = [line.strip() for line in open(melanoma_file, 'r')]

lines = [json.loads(line.strip())+[0] for line in open(benign_stats_file, 'r')]
benign_headers = lines.pop(0)[:-1]+[u"result"]

## add class 1 to atypical nevus
#for name in atypical:
#    for _name in lines:
#        if name in str(_name[4]):
#            _name[12] = 2

benign_df = pd.DataFrame(lines, columns=benign_headers)

# Add class 2 for melanoma
lines = [json.loads(line.strip())+[1] for line in open(melanoma_stats_file,
         'r')]
melanoma_headers = lines.pop(0)[:-1]+[u"result"]
melanoma_df = pd.DataFrame(lines, columns=melanoma_headers)
total_df = pd.concat([benign_df, melanoma_df], axis=0).reset_index()


def train_and_test(percent, total_df):
    global clf
    total_df = total_df.reindex(np.random.permutation(total_df.index))
    columns = ['A1','A2', 'A_B', 'A_BG', 'A_DB', 'A_LB',
               'A_W', 'B', 'C', 'D1', 'D2', 'result']
    total_data = total_df.loc[:,columns].as_matrix()
    np.savetxt('data.txt', total_data, fmt='%d', delimiter=',')

    top = int(float(percent)/100*len(total_data))
    clf.fit(total_data[:top,:-1], total_data[:top,-1])
    # np.savetxt('classifier.txt', clf.coef_, fmt='%10.15f', delimiter=',')
    # print clf.get_params()
    if top < 200:
        print clf.score(total_data[top+1:,:-1],total_data[top+1:,-1])*100
    else:
        print clf.score(total_data[:,:-1],total_data[:,-1])*100

# 25% training dataset and 75% testing dataset
if __name__ == "__main__":
    #clf = svm.LinearSVC()
    clf = neighbors.KNeighborsClassifier(algorithm="brute", weights="distance")
    train_and_test(80, total_df)

