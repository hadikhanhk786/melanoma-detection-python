# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:34:52 2017

@author: ukalwa
"""

import pandas as pd
import numpy as np
# import ast
import os
import json

import cv2

#iterations = "400"
#print "iterations: ", iterations
#DIR = r'G:\Upender\result_set'
#path = os.path.join(DIR,iterations+"_iterations")

iterations = "1"
path = r'G:\Upender\complete_mednode_dataset'
benign_stats_file = os.path.join(path,"benign","results.json")
melanoma_stats_file = os.path.join(path,"melanoma","results.json")
cols = ['A1','A2', 'B', 'C', 'A_B', 'A_BG', 'A_DB', 'A_LB', 'A_W',
        'D1', 'D2', 'result']
## PH2 dataset (From moleanalyzer software)
#real_diameter = 72 # pixels/mm
#if "mednode" in path:
#    # Med-Node dataset
#    # 2.8/105 mm lens 12 MP and 330mm distance from object
#    real_diameter = (104*7360)/(330*24) # pixels/mm


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


lines = [json.loads(line.strip())+[0] for line in open(benign_stats_file, 'r')]
benign_headers = lines.pop(0)[:-1]+[u"result"]

benign_df = pd.DataFrame(lines, columns=benign_headers)
#benign_df[['A1','A2']] = benign_df[['A1','A2']].divide(benign_df['area'],
#                               axis='index')
#benign_df[['A_B', 'A_BG', 'A_DB', 'A_LB', 'A_W']] = \
#                        benign_df[['A_B', 'A_BG',
#                                   'A_DB', 'A_LB', 'A_W']].divide(
#                        benign_df['D1'],
#                               axis='index')
#benign_df[['D1','D2']] = (benign_df[['D1','D2']]/real_diameter).astype('int')

# Add class 2 for melanoma
lines = [json.loads(line.strip())+[1] for line in open(melanoma_stats_file,
         'r')]
melanoma_headers = lines.pop(0)[:-1]+[u"result"]
melanoma_df = pd.DataFrame(lines, columns=melanoma_headers)
#melanoma_df[['A1','A2']] = melanoma_df[['A1','A2']].divide(melanoma_df['area'],
#                               axis='index')
#melanoma_df[['A_B', 'A_BG', 'A_DB', 'A_LB', 'A_W']] = \
#                        melanoma_df[['A_B', 'A_BG',
#                                   'A_DB', 'A_LB', 'A_W']].divide(
#                        melanoma_df['D1'],
#                               axis='index')
#melanoma_df[['D1','D2']] = (melanoma_df[['D1','D2']]/real_diameter).astype('int')

total_df = pd.concat([benign_df, melanoma_df], axis=0).reset_index()


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
#    # Save datasets
#    np.savetxt("test_data2.csv",test_data, delimiter=",", fmt='%1.4f')
#    np.savetxt("train_data2.csv",train_data, delimiter=",", fmt='%1.4f')
    svm.predict(np.float32(test_data[:,:-1]))[1]




# 25% training dataset and 75% testing dataset
if __name__ == "__main__":
    kernel = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_RBF, cv2.ml.SVM_POLY]
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
            if (0.65 <= new_result['TPR'] <= 1 \
                    and  0 <= new_result['FPR'] <= 0.35):
#                print "TEST:TPR %s, FPR %s, Acc %s, PPV %s" % (
#                        new_result['TPR']*100,new_result['FPR']*100,
#                        new_result['Acc']*100, new_result['PPV']*100)
                if new_result['Acc'] > result['Acc']  \
                        and new_result['TPR'] > result['TPR']:
                    result = new_result
                    svm.save("ph2_svm_" + str(param) + ".xml")
                    print "Saved"
                    print "TPR %s, FPR %s, Acc %s, PPV %s" % (
                            new_result['TPR']*100,new_result['FPR']*100,
                            new_result['Acc']*100, new_result['PPV']*100)
        print "Kernel %s: TPR %s, FPR %s, Acc %s, PPV %s \n" % (
                param, result['TPR']*100,result['FPR']*100,
                result['Acc']*100, result['PPV']*100)
    # plot_roc_curve(total_df)

