# -*- coding: utf-8 -*-
"""
This program enables the user to interact with the main program from command
line and GUI. It also allows the user to do batch processing based on the
directory argument.

Created on Wed Jun 07 10:54:17 2017

@author: ukalwa
"""
# Built-in imports
import sys
import argparse
from Tkinter import Tk
import tkFileDialog as filedialog
import os
import json

# third-party imports
import numpy as np

# custom imports
import src.Lesion

# Initialization params
results = []
failed_files = []
save_files = False
performance_metrics = []
iterations = 3
Tk().withdraw()


def main(dir_path='', file_path='', process_dir=False):
    """
    The method that can process a single file or all files in a directory based
    on the arguments provided.

    :param dir_path: Directory path containing all lesion images
    :param file_path: Full file path of a single lesion image
    :param process_dir: If true, it processes all images in a directory
    """
    global failed_files, results, save_files, iterations
    if process_dir:
        if len(dir_path) == 0:
            dir_path = filedialog.askdirectory()

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
                    lesion = src.Lesion.Lesion(filename, iterations=iterations)
                    lesion.extract_info(save=save_files)
                    temp = []
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
                       np.array(performance_metrics) * 1000,
                       delimiter=',')
        else:
            print "Invalid directory"
    else:
        if len(file_path) == 0:
            file_path = filedialog.askopenfilename()
        if len(file_path) != 0:
            print "FILE: %s" % file_path
            lesion = src.Lesion.Lesion(file_path)
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
