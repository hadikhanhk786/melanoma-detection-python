# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:34:52 2017

@author: ukalwa
"""

import pandas as pd
# import numpy as np
import ast

filename = r'G:\Upender\Melanoma Project\PH2Dataset\MoleDetection\results4.txt'
df = pd.DataFrame()
with open(filename, 'r') as data_file:
    for line in data_file.readlines():
        data = ast.literal_eval(line)
        df = df.append(
            pd.DataFrame.from_dict(data, orient='index').transpose(),
            ignore_index=True)
