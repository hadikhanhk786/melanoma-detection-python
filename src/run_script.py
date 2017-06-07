# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 10:54:17 2017

@author: ukalwa
"""
import sys, argparse
import tkFileDialog as filedialog
import os
import posixpath
import json

import Lesion


results = []
failed_files = []
save_files = False

def main(dir_path='', file_path='', process_dir=False):
    global failed_files, results, save_files
    if process_dir:
            if len(dir_path) == 0:
                dir_path = filedialog.askdirectory(
                        initialdir=r'G:\Upender\Melanoma Project\PH2Dataset')

            if len(dir_path) != 0:
                target = open(posixpath.join(dir_path, "results.json"), 'w')
                for name in os.listdir(dir_path):
                    if not os.path.isdir(
                            name) and "lesion" not in name \
                            and "Label" not in name \
                            and "_cropped" in name \
                            and name.endswith('.jpg'):
                        filename = posixpath.join(dir_path, name)
                        print "FILE: %s" % filename
                        lesion = Lesion.Lesion(filename)
                        lesion.extract_info(save=save_files)
                        results.append(lesion.feature_set)
                        target.write(json.dumps(lesion.feature_set) + "\n")
#                        return_vars = process(filename, failed_files, results)
#                        if return_vars is not None:
#                            results.append(return_vars)
                target.close()
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
#            return_vars = process(file_path, failed_files, results)
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
        args = parser.parse_args()
        print args
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
