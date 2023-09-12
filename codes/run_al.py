#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu January 26 15:48:51 2023
Edited on Tues May 02 12:13:40 2023
@author: kangming
@edit: daniel
"""

import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import random

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import math
import time
import pathlib

from distill import get_data, return_model


from uncertaintyAL import grow_random
from uncertaintyAL import grow_QBC
from uncertaintyAL import grow_rfMaxUncertainty
from uncertaintyAL import grow_xgbMaxUncertainty_ibug
print('Dependencies imported successfully!')

#%%
# REQUIRE USER INPUT---------------------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(
    description='Grow datasets with different growing methods.'
    )

parser.add_argument('--growingCriteria', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--outputDir', type=str, required=True)
parser.add_argument('--stepFrac', type=float, required=False, default=0.01)
parser.add_argument('--trainFracStop', type=float, required=False, default=1.0)
parser.add_argument('--randomSeed', type=int, required=False, default=0)


growingCriteria = parser.parse_args().growingCriteria
dataset = parser.parse_args().dataset
target = parser.parse_args().target
outputDir = parser.parse_args().outputDir
fltStepFrac = parser.parse_args().stepFrac
fltTrainFracStop = parser.parse_args().trainFracStop
intRandomSeed = parser.parse_args().randomSeed


# %%
# IMPORT MODELS--------------------------------------------------------------------------------------------------------
# set random state
random_state = 0

# make a dictionary to hold the models
model = {}

# loop through the models and add them to the dictionary
for modelname in ['xgb','rf', 'xgb_1tree']:
    model[modelname] = return_model(modelname, random_state)
    # print(modelname, ' imported successfully: ', model[modelname])

print('Models imported successfully!')

#%%
# GROW DATASETS WITH BASED ON THE ARGUMENTS PASSED---------------------------------------------------------------------

# import data
_ , _, _, xTrainVal, xTest, yTrainVal, yTest = get_data(dataset = dataset, target = target, random_state = 0)
print('\n--------------------------------------------------------------------------------------------')
print ('Dataset: ', dataset, '\nTarget: ', target)
# print the size of the training sets
print('xTrainVal Length: ', len(xTrainVal))
tStart = time.time()

# if the growing criteria is random
if growingCriteria == 'random':
    # grow by random for the dataset and target combination
    grow_random(model['rf'], model['xgb'],
                xTrainVal, yTrainVal,
                xTest, yTest,
                strSaveDir = outputDir,
                strSaveName = dataset + '_' + target + '_random',
                strModel1Name = 'rf',
                strModel2Name = 'xgb',
                fltStepFrac = fltStepFrac,
                fltTrainFracStop = fltTrainFracStop,
                intRandomSeed = intRandomSeed)
    
elif growingCriteria == 'QBC':
    # grow by QBC for the dataset and target combination
    grow_QBC(model['rf'], model['xgb'],
             xTrainVal, yTrainVal,
             xTest, yTest,
             strSaveDir = outputDir,
             strSaveName = dataset + '_' + target + '_QBC',
             strModel1Name = 'rf',
             strModel2Name = 'xgb',
             fltStepFrac = fltStepFrac,
             fltTrainFracStop = fltTrainFracStop,
             intRandomSeed = intRandomSeed)
    
elif growingCriteria == 'rfMaxUncertainty':
    # grow by rfMaxUncertainty for the dataset and target combination
    grow_rfMaxUncertainty(model['rf'], model['xgb'],
                          xTrainVal, yTrainVal,
                          xTest, yTest,
                          strSaveDir = outputDir,
                          strSaveName = dataset + '_' + target + '_rfMaxUncertainty',
                          strModel1Name = 'rf',
                          strModel2Name = 'xgb',
                          fltStepFrac = fltStepFrac,
                          fltTrainFracStop = fltTrainFracStop,
                          intRandomSeed = intRandomSeed)
    
elif growingCriteria == 'xgbMaxUncertainty_ibug':
    # grow by xgbMaxUncertainty_ibug for the dataset and target combination

    grow_xgbMaxUncertainty_ibug(model['xgb_1tree'],
                                model['rf'], model['xgb'],
                                xTrainVal, yTrainVal,
                                xTest, yTest,
                                strSaveDir = outputDir,
                                strSaveName = dataset + '_' + target + '_xgbMaxUncertainty_ibug',
                                strModel1Name = 'rf',
                                strModel2Name = 'xgb',
                                fltStepFrac = fltStepFrac,
                                fltTrainFracStop = fltTrainFracStop,
                                intRandomSeed = intRandomSeed)
else:
    print('Invalid growing criteria!')
    print('Please enter one of the following: random, QBC, rfMaxUncertainty, xgbMaxUncertainty_ibug')
    print('Exiting...')
    exit()   

tEnd = time.time()
print('Time elapsed: ', round(tEnd - tStart, 2), ' seconds')
print ('Done growing by ', growingCriteria, ' on ', dataset, '-', target, '!')
