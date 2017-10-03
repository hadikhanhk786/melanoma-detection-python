# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 10:54:17 2017

@author: ukalwa
"""
import sys, argparse
from Tkinter import Tk
import tkFileDialog as filedialog
import os
import json

import numpy as np

import Lesion


results = []
failed_files = []
save_files = False
performance_metrics = []
iterations = 3
Tk().withdraw()

def main(dir_path='', file_path='', process_dir=False):
    global failed_files, results, save_files, iterations
    if process_dir:
            if len(dir_path) == 0:
                dir_path = filedialog.askdirectory(
                        initialdir=r'G:\Upender\complete_mednode_dataset')

            if len(dir_path) != 0:
                target = open(os.path.join(dir_path, "results.json"), 'w')
                header = []
                notSetHeader = True
                for name in os.listdir(dir_path):
                    if not os.path.isdir(
                            name) and "lesion" not in name \
                            and "Label" not in name \
                            and name.endswith('.jpg'):
                        filename = os.path.join(dir_path, name)
                        print "FILE: %s" % filename
                        lesion = Lesion.Lesion(filename,iterations=iterations)
                        lesion.extract_info(save=save_files)
                        temp = []
#                        search_str = ["A", "B", "C", "D", "image"]
                        for key, value in lesion.feature_set.iteritems():
                            temp.append(value)
                            if notSetHeader:
                                header.append(key)
                        results.append(temp)
                        if notSetHeader:
                            target.write(json.dumps(header) + "\n")
                            notSetHeader = False
                        target.write(json.dumps(temp) + "\n")

                        # Record execution times of different steps
                        performance_metrics.append(lesion.performance_metric)

                target.close()
                # Save the metrics as csv file
                np.savetxt(os.path.join(dir_path, "metrics.csv"),
                           np.array(performance_metrics)*1000,
                           delimiter=',')
            else:
                print "Invalid directory"
    else:
        if len(file_path) == 0:
            file_path = filedialog.askopenfilename(
                    initialdir=r'G:\Upender\Melanoma Project\PH2Dataset')
        if len(file_path) != 0:
            print "FILE: %s" % file_path
            lesion = Lesion.Lesion(file_path)
            lesion.extract_info(save=save_files)
        else:
            print "Invalid file"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "no arguments passed"
        main(process_dir=False)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--dir", help="directory containing files")
        parser.add_argument("-f", "--file", help="full file path")
        parser.add_argument("--save", help="save Images")
        parser.add_argument("-i", "--iterations", help="Iterations(0--3)")
        args = parser.parse_args()
        # print args
        if args.save:
            save_files = True
        if args.iterations:
            print "iterations:", args.iterations
            iterations = args.iterations
        if args.dir:
            if os.path.isdir(args.dir):
                main(dir_path=args.dir, process_dir=True)
            else:
                print "Invalid directory"
                main(process_dir=True)
        elif args.file:
            if os.path.isfile(args.file):
                main(file_path=args.file, process_dir=False)
            else:
                print "Invalid file"
                main(process_dir=False)
