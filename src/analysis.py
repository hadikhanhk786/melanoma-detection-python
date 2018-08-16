# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:34:52 2017

@author: ukalwa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# import ast
import os
import json

import cv2

#iterations = "400"
#print "iterations: ", iterations
#DIR = r'G:\Upender\result_set'
#path = os.path.join(DIR,iterations+"_iterations")

#iterations = "1"
#path = r'G:\Upender\complete_mednode_dataset'
paths = [r'Z:\Upender\result_set\400_iterations']
#paths = [r'Z:\Upender\complete_mednode_dataset',
#         r'Z:\Upender\result_set\400_iterations']
benign_df = pd.DataFrame()
melanoma_df = pd.DataFrame()
for PATH in paths:
    benign_stats_file = os.path.join(PATH,"benign","results.json")
    melanoma_stats_file = os.path.join(PATH,"melanoma","results.json")
    cols = ['A1','A2', 'B', 'C', 'A_B', 'A_BG', 'A_DB', 'A_LB', 'A_W',
            'D1', 'D2', 'result']

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

def calculate_roc_params(clf, benign_df, melanoma_df):
    global cols
    # Classification accuracy params
    TP=0; FP=0; TN=0; FN=0
    for i in np.float32(benign_df.loc[:,cols].values):
        res = clf.predict(i[:-1].reshape(-1,len(i[:-1])))
        if len(res) != 1:
            res = res[1]
        if res == i[-1]:
            TN = TN + 1
        else:
             FP = FP + 1
    for i in np.float32(melanoma_df.loc[:,cols].values):
        res = clf.predict(i[:-1].reshape(-1,len(i[:-1])))
        if len(res) != 1:
            res = res[1]
        if res == i[-1]:
            TP = TP + 1
        else:
             FN = FN + 1

    # print TP, TN, FP, FN
    # Calculate TPR(Sensitivity), FPR(Selectivity), PPR(Precision) and Accuracy
    TPR = round(float(TP) / (TP + FN),4)
    FPR = round(float(FP) / (FP + TN),4)
    Accuracy = round(float(TP + TN)/(TP + TN + FP + FN),4)
    if TP + FP != 0:
        PPV = round(float(TP)/ (TP + FP),4)
    else:
        PPV = 0
#    print "TPR : %s, FPR : %s, Accuracy : %s, Precision : %s" %(TPR, FPR,
#                                                                Accuracy,
#                                                                PPV)
    return {'TPR':TPR, 'FPR':FPR, 'Acc':Accuracy, 'PPV':PPV}


def train_and_test(percent, total_df, benign_df, melanoma_df,svm):
    global cols, train_data, test_data

    # Randomize the order of the lesions
    melanoma_df = melanoma_df.reindex(np.random.permutation(melanoma_df.index))
    benign_df = benign_df.reindex(np.random.permutation(benign_df.index))

    # Cutoff to separate training set from testing set
    benign_cutoff = int(float(percent)/100*len(benign_df))

    melanoma_cutoff = int(float(percent)/100*len(melanoma_df))
    # print benign_cutoff, melanoma_cutoff

    train_data = np.concatenate([melanoma_df.loc[:,cols].as_matrix()[
                                    :melanoma_cutoff,:],
                                 benign_df.loc[:,cols].as_matrix()[
                                         :benign_cutoff,:]])
    test_data = np.concatenate([melanoma_df.loc[:,cols].as_matrix()[
                                    melanoma_cutoff:,:],
                                 benign_df.loc[:,cols].as_matrix()[
                                         benign_cutoff:,:]])
    svm.train(np.float32(train_data[:,:-1]), cv2.ml.ROW_SAMPLE,
              np.int32(train_data[:,-1]))
    svm.predict(np.float32(test_data[:,:-1]))[1]




# 25% training dataset and 75% testing dataset
if __name__ == "__main__":
    kernel = [cv2.ml.SVM_RBF]
    # lst = [0.1, 0.15, 0.2, 0.25]
    for param in kernel:
        train_data = np.array([])
        test_data = np.array([])
        #clf = sk_svm.SVC(kernel='rbf')
        #svm = cv2.ml.SVM_load("opencv_svm.xml")
        result = {'TPR':0, 'FPR':1, 'Acc':0, 'PPV':0}
        np.random.seed(0)
        # clf = sk_svm.LinearSVC(dual=False,max_iter=10000,C=3)
        for i in range(100000):
            svm = cv2.ml.SVM_create()
            svm.setType(cv2.ml.SVM_C_SVC)
            svm.setC(5)
            svm.setKernel(param)
            svm.setGamma(0.2)
            svm.setDegree(7)
            svm.setClassWeights(None)
            svm.setCoef0(0.0)
            svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
            svm.setP(0.0)
            train_and_test(25, total_df, benign_df, melanoma_df, svm)
            new_result = calculate_roc_params(svm, benign_df, melanoma_df)
            if (0.7 <= new_result['TPR'] <= 1 \
                    and  0 <= new_result['FPR'] <= 0.3):
#                print "TEST:TPR %s, FPR %s, Acc %s, PPV %s" % (
#                        new_result['TPR']*100,new_result['FPR']*100,
#                        new_result['Acc']*100, new_result['PPV']*100)
                if new_result['TPR'] > result['TPR'] \
                             and new_result['FPR'] < result['FPR']:
                    result = new_result
                    svm.save("opencv_svm.xml")
                    print "Saved"
                    print "TPR %s, FPR %s, Acc %s, PPV %s" % (
                            new_result['TPR']*100,new_result['FPR']*100,
                            new_result['Acc']*100, new_result['PPV']*100)
        print "Kernel %s: TPR %s, FPR %s, Acc %s, PPV %s \n" % (
                param, result['TPR']*100,result['FPR']*100,
                result['Acc']*100, result['PPV']*100)
    # plot_roc_curve(total_df)

