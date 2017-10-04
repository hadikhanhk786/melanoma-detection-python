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

PATH = r'G:\Upender\result_set'
folders = ['50_iterations', '100_iterations', '200_iterations', '400_iterations']
dirs = []
for folder in folders:
    dirs.append(os.path.join(PATH,folder))

dirs.append(r'G:\Upender\complete_mednode_dataset')
model_name = ['opencv_svm_50iter.xml', 'opencv_svm_100iter.xml',
              'opencv_svm_200iter.xml', 'opencv_svm.xml']
cols = ['A1','A2', 'B', 'C', 'A_B', 'A_BG', 'A_DB', 'A_LB', 'A_W',
        'D1', 'D2', 'result']

## PH2 dataset (From moleanalyzer software)
#real_diameter = 72 # pixels/mm
#if "mednode" in PATH:
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
    # Calculate TPR(Sensitivity), TNR(Selectivity), PPR(Precision) and Accuracy
    TPR = float(TP) / (TP + FN)
    TNR = float(TN) / (FP + TN)
    Accuracy = float(TP + TN)/(TP + TN + FP + FN)
#    PPV = float(TP)/ (TP + FP)
#    print "TPR : %s, FPR : %s, Accuracy : %s, Precision : %s" %(TPR, TNR,
#                                                                Accuracy,
#                                                                PPV)
    print "%s,%1.2f,%s" % (TPR*100, TNR*100, Accuracy*100)





# 25% training dataset and 75% testing dataset
if __name__ == "__main__":
    #clf = sk_svm.SVC(kernel='rbf')
    for xmlfile in model_name:
        svm = cv2.ml.SVM_load(xmlfile)
        print xmlfile
        for dirname in dirs:
            dir_PATH = os.path.join(PATH,dirname)
#            print dir_PATH
            benign_stats_file = os.path.join(dir_PATH,"benign",
                                             "results.json")
            melanoma_stats_file = os.path.join(dir_PATH,"melanoma",
                                               "results.json")
            lines = [json.loads(line.strip())+[0] for line in \
                     open(benign_stats_file, 'r')]
            benign_headers = lines.pop(0)[:-1]+[u"result"]

            benign_df = pd.DataFrame(lines, columns=benign_headers)

            # Add class 2 for melanoma
            lines = [json.loads(line.strip())+[1] for line in open(
                    melanoma_stats_file,'r')]
            melanoma_headers = lines.pop(0)[:-1]+[u"result"]
            melanoma_df = pd.DataFrame(lines, columns=melanoma_headers)


            total_df = pd.concat([benign_df, melanoma_df], axis=0
                                 ).reset_index()

            #Test results
            calculate_roc_params(svm, benign_df, melanoma_df)


    # plot_roc_curve(total_df)

