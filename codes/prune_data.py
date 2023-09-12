#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:17:47 2022

@author: kangming
"""

import os
import json
import pandas as pd
# import pyarrow
import pickle

import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import math
import time
import pathlib  

from distill import get_data,get_scores_random,return_model,prune_rd
from myfunc import get_scores

random_state = 0
test_size = 0.1


model = {}
for modelname in ['xgb','rf']:
    model[modelname] = return_model(modelname,random_state)


#%%
'''
Get MAD and STD
'''
for target in ['e_form','bandgap']:
    for dataset in ['jarvis22']: #'jarvis','jarvis22','mp','oqmd1.0','oqmd'
        df, X, y, X_train_val, X_test, y_train_val, y_test = get_data(
            dataset, target = target, random_state = random_state)
        mad = (y - y.mean()).abs().mean()
        std = y.std()
        print('')
        print('-----------------')
        print(f'{target} {dataset} {y.shape[0]} {mad:.3f} {std:.3f} ')
        print('-----------------')
        print('')


            
#%%
'''
Random baseline
'''
target_list = ['e_form'] #+ ['log10_bulk_modulus','log10_shear_modulus']

# list of dataset name
dataset_list =  ['jarvis22'] # ['jarvis','jarvis22', 'mp', 'mp18', 'oqmd1.0']
# list of model names
model_list = ['xgb','rf']
# number of independent random sampling
n_rand = {'xgb':10, 'rf':10, 'alignn': 3}

target2transfer = None #'bulk_modulus'

folder='random'
pathlib.Path(f"./{folder}").mkdir(parents=True, exist_ok=True)

# # List of training set size
train_size_list = {}
train_size_list['xgb'] = (
        [i/1000 for i in range(1,10)] 
        + [i/100 for i in range(1,10)] 
        + [i/100 for i in range(10,30,5)] 
        + [i/100 for i in range(30,100,10)]
        )
train_size_list['rf'] = train_size_list['xgb']

for target in target_list:# 'e_form'    
    for dataset in dataset_list:     
        
        print(f'Start reading dataset {dataset}')
        
        col_X = None
            
        _, _, _ , X_train_val, X_test, y_train_val, y_test = get_data(
            dataset, target = target, random_state = random_state,
            target2transfer = target2transfer,
            col_X = col_X
            )            
        for modelname in model_list:
            for random_state in range(n_rand[modelname]):
                if target2transfer is None:
                    csv_out = f'{folder}/{target}_{dataset}_{modelname}_{random_state}.csv'
                else:
                    csv_out = f'{folder}/{target}_for_{target2transfer}_{dataset}_{modelname}_{random_state}.csv'
                
                # if pathlib.Path(csv_out).is_file():
                #     print(f'{csv_out} found, skipping the random sampling')
                # else:
                if True:
                    if modelname == 'alignn':
                        model['alignn'].config.target = target
                    
                    get_scores_random(
                            model[modelname],
                            X_train_val, y_train_val, train_size_list[modelname], random_state,
                            X_test, y_test,
                            csv_out,
                            mode = 'regression'
                            )
    
    
#%%
'''
pruning 

'''


reprune = True

target_list = ['e_form',] #+,'bandgap' ['log10_bulk_modulus','log10_shear_modulus']
target2transfer = None #'bulk_modulus'

for dataset in ['jarvis22']:
    for target in target_list:
        
        if target == 'bulk_modulus' and (dataset == "oqmd1.0" or dataset=='oqmd'):
            continue
        
        df, _, _ , X_train_val, X_test, y_train_val, y_test = get_data(
            dataset, target = target, random_state = random_state,
            target2transfer = target2transfer
            )
        
        '''
        xgb guiding
        '''
        
        guiding = 'xgb'
        guided = 'rf'
        
        if target2transfer is None:
            folder=f'{target}/{target}_{dataset}_{guiding}_guiding_pruning'
        else:
            folder=f'{target}/{target}_for_{target2transfer}_{dataset}_{guiding}_guiding_pruning'
        pathlib.Path(f"./{folder}").mkdir(parents=True, exist_ok=True)
        file_out = f'{folder}/all_dat.pkl'
        
        if pathlib.Path(file_out).is_file() and not reprune:
            print(f'{file_out} found, skipping...')
        else:
            if pathlib.Path(file_out).is_file(): # meaning reprune set to True
                print(f'{file_out} found, but forcing repruning...')
            else:
                print(f'{file_out} NOT found, pruning...')
                
            # Full dataset
            maes, rmse, r2 = get_scores(
                model[guiding],X_train_val,y_train_val,X_test,y_test
                )
            # Pruning dataset
            size_old_val,ids,test_scores,val_scores = prune_rd(
                model[guiding],X_train_val,y_train_val,X_test,y_test,
                model_test=model[guided],
                min_drop = int(X_train_val.shape[0]/100),
                threshold=maes/2, train_size=0.8, file_out=file_out
                )
            
        
        '''
        rf guiding
        '''     
        
        guiding = 'rf'
        guided = 'xgb'
        
        folder=f'{target}/{target}_{dataset}_{guiding}_guiding_pruning'
        pathlib.Path(f"./{folder}").mkdir(parents=True, exist_ok=True)
        file_out = f'{folder}/all_dat.pkl'
        
        if pathlib.Path(file_out).is_file() and not reprune:
            print(f'{file_out} found, skipping...')
        else:
            if pathlib.Path(file_out).is_file(): # meaning reprune set to True
                print(f'{file_out} found, but forcing repruning...')
            else:
                print(f'{file_out} NOT found, pruning...')      
            # Full dataset
            maes, rmse, r2 = get_scores(model[guiding],X_train_val,y_train_val,X_test,y_test)
            # Pruning dataset
            size_old_val,ids,test_scores,val_scores = prune_rd(
                model[guiding],X_train_val,y_train_val,X_test,y_test,
                model_test=model[guided],
                min_drop = int(X_train_val.shape[0]/100),
                threshold=maes/2,train_size=0.8, file_out=file_out
                )
