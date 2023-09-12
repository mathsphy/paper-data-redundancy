#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
from myfunc import get_scores,drop_failed_structures

from glob import glob

#%%

def return_model(modelname,random_state):
    
    xgb_tree_method = 'gpu_hist'
    
    model = {}
    if modelname == 'xgb':
        return xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.25,
        reg_lambda=0.01,reg_alpha=0.1,
        subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
        num_parallel_tree=4 ,tree_method=xgb_tree_method
        )
    
    if modelname == 'xgb_1tree':
        return xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.25,
        reg_lambda=0.01, #reg_alpha=0.1, # ibug requires that model_params['reg_alpha'] == 0
        # subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
        num_parallel_tree=1 ,tree_method=xgb_tree_method
        )

    elif modelname == 'xgb_cla':
        return xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.25,
        reg_lambda=0.01,reg_alpha=0.1,
        subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
        num_parallel_tree=4,tree_method=xgb_tree_method
        )
    elif modelname == 'rf':
        return RandomForestRegressor(
        n_estimators=100, max_features=1/3, n_jobs=-1, random_state=random_state
        )
    
    elif modelname == 'alignn':
        try:
            from jarvis.db.jsonutils import loadjson
            from alignn.config import TrainingConfig
            from sklearnutils import AlignnLayerNorm
            config_filename = 'config.json'
            config = loadjson(config_filename)
            config = TrainingConfig(**config)
            return AlignnLayerNorm(config)    
        except Exception as e: 
            print(e)
        
    elif modelname == 'xgb_bandgap':
        class bandgap_classifier_regressor:
            def __init__(self, classifier, regressor):
                self.classifier = classifier
                self.regressor = regressor
                
            def fit(self, X, y):
                # true if non-metal
                y_cla = (y!=0)
                self.classifier.fit(X,y_cla)
                y_reg = y[y>0]
                X_reg = X[y>0]
                self.regressor.fit(X_reg, y_reg)
                
            def predict(self, X):
                y_pred = pd.Series(self.classifier.predict(X).astype(bool),index=X.index)
                nonmetal = y_pred[y_pred].index
                metal = y_pred[~y_pred].index
                # set bandgap of metal to zero
                y_pred.loc[metal] = 0
                # use regressor to get bandgap of non-metal
                y_pred.loc[nonmetal] = self.regressor.predict(X.loc[nonmetal])
                return y_pred
            
            
        return bandgap_classifier_regressor(
            xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.25,
            reg_lambda=0.01,reg_alpha=0.1,
            subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
            num_parallel_tree=4,tree_method=xgb_tree_method
                ),
            xgb.XGBRegressor(
                n_estimators=1000, learning_rate=0.25,
                reg_lambda=0.01,reg_alpha=0.1,
                subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
                num_parallel_tree=4,tree_method=xgb_tree_method
                )
            )
    else:
        raise ValueError(f'Unknown model: {modelname}')


#%%

'''
Define the function that returns the X and y
'''

def get_data(
        dataset,
        test_size=0.1,
        random_state=0,
        target='e_form', 
        fixed_train_ids=None,
        target2transfer=None,
        get_pruned_set=False,
        pruning_model=None,
        pruned_set_min_frac=0.3,
        col_X = None,
        standardized=True,
        data_dir = 'data',
        predefined_train_val_test = False,
        ):

    if col_X is None:
        df = pd.read_pickle(f'{data_dir}/{dataset}/dat_featurized_matminer.pkl')
    elif col_X == 'precomputed_graphs':
        df = pd.read_pickle(f'{data_dir}/{dataset}/dat_featurized.pkl')
    # else:
    #     df = pd.read_pickle(f'data/{dataset}/dat_featurized_alignn.pkl')

    
    df = df.replace([np.inf, -np.inf], np.nan)
    if target2transfer is None:
        df = df[~df[target].isna()]
    else:
        df = df[~df[target2transfer].isna()]
    df = drop_failed_structures(df)
    
    # Remove structures with e_form > 5 eV/atom
    if 'e_form' in df.columns:
        max_Ef = 5
        df = df[df['e_form']<=max_Ef]   
       

    # Get standardized X, and y
    if col_X is None:
        X_no_std = df.iloc[:,-273:]
        if standardized:
            X = pd.DataFrame(
                StandardScaler().fit_transform(X_no_std),
                index=X_no_std.index, columns=X_no_std.columns
                )
        else:
            X = X_no_std
    else:
        X = df[col_X]
    
    
    y = df[target]

    if get_pruned_set:
        guiding = pruning_model
        folder=f'{target}/{target}_{dataset}_{guiding}_guiding_pruning'
        reformat_results(folder)  
        with open(f'{folder}/all_dat.pkl','rb') as f:
            [size_old_val,ids,test_scores,val_scores] = pickle.load(f)
        n_data = int(pruned_set_min_frac * len(ids['train_val']))
        ids_to_use = ids['train_val'][:n_data]
        X = X.loc[ids_to_use]
        y = y.loc[ids_to_use]


    # Random train-val-test split
    X_val, y_val = None, None
    if fixed_train_ids is None:
        if predefined_train_val_test:
            # load json file
            with open(f'{data_dir}/{dataset}/train_val_test.json','r') as f:
                train_val_test = json.load(f)
                train = list(train_val_test["train"].keys())
                val = list(train_val_test["val"].keys())
                # val = []
                test = list(train_val_test["test"].keys())
                # # check if val equal to test
                # if set(val) == set(test):
                #     print('Warning: val and test are the same')
                #     print('Setting val to empty')
                #     val = []

            X_pool = X.loc[train]
            y_pool = y.loc[train]
            X_val = X.loc[val]
            y_val = y.loc[val]
            X_test = X.loc[test]
            y_test = y.loc[test]
            print(f'Size of the training set: {X_pool.shape[0]}')
            print(f'Size of the validation set: {X_val.shape[0]}')
            print(f'Size of the test set: {X_test.shape[0]}')
        else:
            X_pool, X_test, y_pool, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
                )
    else:
        '''
        fixed_train_ids are the ids of entries that will always be kept in the training set
        
        '''
        
        X_fixed_train = X.loc[fixed_train_ids]
        y_fixed_train = y.loc[fixed_train_ids]
        X_pool, X_test, y_pool, y_test = train_test_split(
            X.drop(fixed_train_ids), y.drop(fixed_train_ids), 
            test_size=test_size, random_state=random_state
            ) 
        print(f'Size of the fixed training set: {X_fixed_train.shape[0]}')
        print(f'Size of the pool: {X.drop(fixed_train_ids).shape[0]}')


    if fixed_train_ids is None:
        if X_val is None:
            return df, X, y , X_pool, X_test, y_pool, y_test
        else:
            return df, X, y , X_pool, X_test, y_pool, y_test, X_val, y_val        
    
    else:
        return df, X, y , X_pool, X_test, y_pool, y_test, X_fixed_train, y_fixed_train


#%%
'''
Function performing random baseline policy
'''

def get_scores_random(
        model,
        X_pool, y_pool, train_size_list, random_state,
        X_test, y_test,
        csv_out = None,
        mode = 'regression',
        random_shuffle = True
        ):  
    
    scores = []  
    for train_size in train_size_list:
        if random_shuffle:
            X_train, X_val, y_train, y_val = train_test_split(
                X_pool, y_pool, train_size=train_size, random_state=random_state
                )
        else:
            print('Warning: No shuffle')
            ts = int(train_size * X_pool.shape[0])
            X_train, X_val = X_pool.iloc[:ts], X_pool.iloc[ts:]
            y_train, y_val = y_pool.iloc[:ts], y_pool.iloc[ts:]
        print('')
        print(f'train_size={train_size}')
        if mode == 'regression':
            maes, rmse, r2, maes_val, rmse_val, r2_val = get_scores(
                model,X_train,y_train,X_test,y_test,X_val,y_val
                )
            scores.append([train_size, maes, rmse, r2, maes_val, rmse_val, r2_val])
        print('')
    # Convert to df and save to csv
    scores = pd.DataFrame(
        scores,
        columns=['train_size','maes','rmse','r2','maes_val','rmse_val','r2_val']
        ).set_index('train_size')    
    if csv_out is not None:
        scores.to_csv(csv_out)

#%%

def prune_rd(
        model,X_pool,y_pool,X_test,y_test,threshold,train_size,
        file_out,
        min_drop=None,max_iter=None,drop_max_err=None,model_test=None,
        join_model=False,threshold2=None,
        X_fixed_train=None, y_fixed_train=None,
        retrain=True,
        autorestart=True
        # threshold_take_back = None
           ):
    '''
    Prune redundant data

    Parameters
    ----------
    model : sklearn model to train
    
    X_pool, y_pool : features and labels (dataframe) of the dataset 
        excluding the hold-out test set
    
    X_test, y_test : features and labels (dataframe) of the hold-out test set
    
    threshold : the data with prediction errors above this threshold will be pruned
    
    train_size : Used in the train-test split of the training pool in each iteration
        
    min_drop : minimal number of samples to drop in each iteration
        
    max_iter : TYPE, optional
        DESCRIPTION. The default is None.
        
    drop_max_err : TYPE, optional
        DESCRIPTION. The default is None.
        
    model_test : TYPE, optional
        DESCRIPTION. The default is None.
        
    X_fixed_train : TYPE, optional
        DESCRIPTION. The default is None.
    y_fixed_train : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    size_old_val : TYPE
        DESCRIPTION.
    ids : TYPE
        DESCRIPTION.
    test_scores : TYPE
        DESCRIPTION.
    val_scores : TYPE
        DESCRIPTION.

    '''
    

    i_initial=0
    # Initialize
    ids={}
    ids['train_val'] = X_pool.index.tolist() # pool
    ids['old_val'] = [] # ALL data that are removed from training
    size_old_val = []
    ids_to_remove = [] # data which are removed from training in the current iteration
    test_scores={'maes':[], 'rmse':[], 'r2':[], 'maes_m2':[], 'rmse_m2':[], 'r2_m2':[]}
    val_scores={'maes':[], 'rmse':[], 'r2':[], 'maes_m2':[], 'rmse_m2':[], 'r2_m2':[]}
    

    # continue from previous pruning
    
    if pathlib.Path(file_out+'.tmp').is_file() and autorestart:
        print(file_out+'.tmp'+' found.')
        with open(file_out+'.tmp','rb') as f:
            [size_old_val,ids,test_scores_in,val_scores_in] = pickle.load(f)
        
        if isinstance(test_scores_in, pd.DataFrame):
            for col in test_scores_in.columns:
                test_scores[col] = test_scores_in[col].tolist()

        if isinstance(val_scores_in, pd.DataFrame):
            for col in val_scores_in.columns:
                val_scores[col] = val_scores_in[col].tolist()            
                
        print('Continue from previous pruning.')
        i_initial = len(test_scores['r2'])
        
        X = pd.concat([X_pool,X_test])
        y = pd.concat([y_pool,y_test])
        X_pool = X.loc[ids['train_val']]
        y_pool = y.loc[ids['train_val']]
        test_ids = list(set(X.index.tolist()) - set(ids['train_val']))
        X_test = X.loc[test_ids]        
        y_test = y.loc[test_ids]
        print('Resetting train_val and test:')
        print(f'X_pool shape: {X_pool.shape}')
        print(f'X_test shape: {X_test.shape}')

        
    
    if min_drop is None:
        min_drop = int(X_pool.shape[0]/100)
        
    if max_iter is None:
        max_iter = 2000    
    
    for i in range(i_initial,max_iter):
        # Save a temp file
        if i > i_initial+1:
            with open(file_out+'.tmp','wb') as f:
                pickle.dump([size_old_val,ids,test_scores,val_scores],f)
            
        ids['old_val'].extend(ids_to_remove)           
        ids['train_new_val'] = list(set(ids['train_val']) - set(ids['old_val']))
        
        X_train_new_val = X_pool.loc[ids['train_new_val']]
        y_train_new_val = y_pool.loc[ids['train_new_val']]
        # Split
        X_train, X_new_val, y_train, y_new_val = train_test_split(
            X_train_new_val, y_train_new_val, train_size=train_size, random_state=i
            )
        start_time = time.time()
        # Train
        if X_fixed_train is None:
            model.fit(X_train,y_train)
        else:
            model.fit(
                pd.concat([X_train,X_fixed_train]),
                pd.concat([y_train,y_fixed_train])
                )
        # predict and get the abs errors 
        y_err_new_val = (model.predict(X_new_val) - y_new_val).abs().sort_values()
        # drop by threshold
        ids_to_remove = y_err_new_val[y_err_new_val<threshold].index.tolist()  
        
        if join_model:
            ''' Here I simply replace model by model_test '''
            if X_fixed_train is None:
                model_test.fit(X_train,y_train)
            else:
                model_test.fit(
                    pd.concat([X_train,X_fixed_train]),
                    pd.concat([y_train,y_fixed_train])
                    )
            # predict and get the abs errors 
            y_err_new_val = (model_test.predict(X_new_val) - y_new_val).abs().sort_values()
            # drop by threshold
            ''' threshold2, and intersection '''
            ids_to_remove = list(
                set(ids_to_remove) & set(y_err_new_val[y_err_new_val<threshold2].index.tolist())
                )            
        
        # min number of drop
        if min_drop is not None:
            if len(ids_to_remove) < min_drop:
                ids_to_remove = y_err_new_val.iloc[:min_drop].index.tolist()
        # whether drop samples with max errors
        if drop_max_err is not None:
            ids_to_remove.extend(y_err_new_val.iloc[-drop_max_err:].index.tolist())
        
        '''
        Get the sizes
        '''             
        size_train_new_val = len(ids['train_new_val'])
        size_to_remove = len(ids_to_remove)
        size_all = X_pool.shape[0]
        
        # return
        if size_to_remove > size_train_new_val-300:
            with open(file_out,'wb') as f:
                pickle.dump([size_old_val,ids,test_scores,val_scores],f)
            print(f'Stopping pruning. size_to_remove={size_to_remove},'+
                  f'min_drop={min_drop}, size_train_new_val={size_train_new_val}')
            return size_old_val,ids,test_scores,val_scores  

        size_old_val.append(len(ids['old_val']))
        print('')
        print('')
        print(f'------ Iteration {i} ------')
        print('')
        print(f'To remove: {size_to_remove}')
        print(f'Current old_val:{size_old_val[-1]} (ratio= {size_old_val[-1]/size_all:.3f})')
        print(f'Current train_new_val: {size_train_new_val} (ratio= {size_train_new_val/size_all:.3f})')
        print("--- %s seconds ---" % (time.time() - start_time))
 

        '''
        Get test and val scores for the main model
        '''
        start_time = time.time()
        # test scores
        if retrain:
            if X_fixed_train is None:
                model.fit(X_train_new_val,y_train_new_val)
            else:
                model.fit(
                    pd.concat([X_train_new_val,X_fixed_train]),
                    pd.concat([y_train_new_val,y_fixed_train])
                    )
                
        y_pred = model.predict(X_test)        
        maes = metrics.mean_absolute_error(y_test,y_pred)
        rmse = metrics.mean_squared_error(y_test,y_pred,squared=False)
        r2 = metrics.r2_score(y_test,y_pred)
        print('')
        print(f'Test scores: maes={maes:.3f}, rmse={rmse:.3f}, r2={r2:.3f}')
        test_scores['maes'].append(maes)
        test_scores['rmse'].append(rmse)
        test_scores['r2'].append(r2)
        # val scores
        if i > 0:
            y_pred = model.predict(X_pool.loc[ids['old_val']])      
            y_old_val = y_pool.loc[ids['old_val']]
            maes = metrics.mean_absolute_error(y_old_val,y_pred)
            rmse =metrics.mean_squared_error(y_old_val,y_pred,squared=False) 
            r2 = metrics.r2_score(y_old_val,y_pred)
            print(f'Val scores: maes={maes:.3f}, rmse={rmse:.3f}, r2={r2:.3f}')
            val_scores['maes'].append(maes)
            val_scores['rmse'].append(rmse)
            val_scores['r2'].append(r2)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        '''
        Get test and val scores for the 2nd model (used to check transferability)
        '''            
        if model_test is not None:
            # test scores
            start_time = time.time()
            if X_fixed_train is None:
                model_test.fit(X_train_new_val,y_train_new_val)
            else:
                model_test.fit(
                    pd.concat([X_train_new_val,X_fixed_train]),
                    pd.concat([y_train_new_val,y_fixed_train])
                    )
                
            y_pred = model_test.predict(X_test)        
            maes = metrics.mean_absolute_error(y_test,y_pred)
            rmse = metrics.mean_squared_error(y_test,y_pred,squared=False)
            r2 = metrics.r2_score(y_test,y_pred)
            print('')
            print(f'Test scores: maes={maes:.3f}, rmse={rmse:.3f}, r2={r2:.3f}')
            test_scores['maes_m2'].append(maes)
            test_scores['rmse_m2'].append(rmse)
            test_scores['r2_m2'].append(r2)
            # val scores
            if i > 0:
                y_pred = model_test.predict(X_pool.loc[ids['old_val']])      
                y_old_val = y_pool.loc[ids['old_val']]
                maes = metrics.mean_absolute_error(y_old_val,y_pred)
                rmse =metrics.mean_squared_error(y_old_val,y_pred,squared=False) 
                r2 = metrics.r2_score(y_old_val,y_pred)
                print(f'Val scores: maes={maes:.3f}, rmse={rmse:.3f}, r2={r2:.3f}')
                val_scores['maes_m2'].append(maes)
                val_scores['rmse_m2'].append(rmse)
                val_scores['r2_m2'].append(r2)  
            print("--- %s seconds ---" % (time.time() - start_time))
        
    with open(file_out,'wb') as f:
        pickle.dump([size_old_val,ids,test_scores,val_scores],f)
    return size_old_val,ids,test_scores,val_scores




    


#%%
def csv2scores_random(target,dataset,modelname,get_val=False):
    
    prefix = f'random/{target}_{dataset}_{modelname}_'
    suffix = '.csv'
    filelist = glob(prefix+'*'+suffix)
    filelist.sort()
    # print(f'{len(filelist)} csv files found for {target}_{dataset}_{modelname}')
    
    scores = {}    
    for score in ['maes','rmse','r2','maes_val','rmse_val','r2_val']:
        scores[score]=pd.DataFrame()
        
        for file_random in filelist:
            n = file_random.replace(prefix,'').replace(suffix,'')
            scores[score][n] = pd.read_csv(file_random,index_col=0)[score]
    
    if get_val:
        scores = pd.DataFrame(
            data = {
                'maes': scores['maes_val'].mean(axis=1),
                'maes_std': scores['maes_val'].std(axis=1),
                'rmse': scores['rmse_val'].mean(axis=1),
                'rmse_std': scores['rmse_val'].std(axis=1),
                'r2': scores['r2_val'].mean(axis=1),
                'r2_std': scores['r2_val'].std(axis=1)
                }
            )
    else:
        scores = pd.DataFrame(
            data = {
                'maes': scores['maes'].mean(axis=1),
                'maes_std': scores['maes'].std(axis=1),
                'rmse': scores['rmse'].mean(axis=1),
                'rmse_std': scores['rmse'].std(axis=1),
                'r2': scores['r2'].mean(axis=1),
                'r2_std': scores['r2'].std(axis=1)
                }
            )
    
    return scores




def reformat_results(folder):

    def fill_None(test_scores):
        if isinstance(test_scores, dict):
            null_keys = [i for i in test_scores.keys() if len(test_scores[i])==0]        
            nonnull_keys = [i for i in test_scores.keys() if len(test_scores[i])!=0]
            test_scores = pd.DataFrame({i: test_scores[i] for i in nonnull_keys})
            for key in null_keys:
                test_scores[key]=None
        return test_scores
    
    if pathlib.Path(f'{folder}/all_dat.pkl').is_file():        
        with open(f'{folder}/all_dat.pkl','rb') as f:
            [size_old_val,ids,test_scores,val_scores] = pickle.load(f)
    elif pathlib.Path(f'{folder}/all_dat.pkl.tmp').is_file():        
        with open(f'{folder}/all_dat.pkl.tmp','rb') as f:
            [size_old_val,ids,test_scores,val_scores] = pickle.load(f)
    
    else:
        with open(f'{folder}/val_scores.pkl','rb') as f:
            val_scores = pickle.load(f)  
            
        with open(f'{folder}/test_scores.pkl','rb') as f:
            test_scores = pickle.load(f)
    
        with open(f'{folder}/ids.pkl','rb') as f:
            ids = pickle.load(f)
    
        with open(f'{folder}/size_old_val.pkl','rb') as f:
            size_old_val = pickle.load(f)
        

    
    dat_size = pd.DataFrame(size_old_val)
    dat_size.columns = ['val_size']
    tot_train_val_size = len(ids['train_val'])
    dat_size['val_ratio'] = dat_size['val_size']/tot_train_val_size
    dat_size['train_size'] =  tot_train_val_size - dat_size['val_size']
    dat_size['train_ratio'] = 1 - dat_size['val_ratio']
    test_scores['train_ratio'] = dat_size['train_ratio'] #.drop_duplicates()
    val_scores['train_ratio'] = dat_size['train_ratio'] #.drop_duplicates()
    
    # -- Fix a mini bug in my previous code ----
    diff = len(test_scores['train_ratio']) - len(test_scores['r2'])
    if diff >0:
        test_scores['train_ratio'] = test_scores['train_ratio'][:-diff]
        # with open(f'{folder}/all_dat.pkl'+'.tmp','wb') as f:
        #     pickle.dump([size_old_val,ids,test_scores,val_scores],f)
            
    diff = len(val_scores['train_ratio']) - len(val_scores['r2'])
    if diff >0:
        val_scores['train_ratio'] = val_scores['train_ratio'][:-diff]
    #     with open(f'{folder}/all_dat.pkl'+'.tmp','wb') as f:
    #         pickle.dump([size_old_val,ids,test_scores,val_scores],f)  
            
    # ------------------------------------------    
    
    test_scores = fill_None(test_scores)
    val_scores = fill_None(val_scores)

    ids['train_val'] = (ids['old_val'] + ids['train_new_val'])
    ids['train_val'].reverse()
    
    with open(f'{folder}/all_dat.pkl','wb') as f:
        pickle.dump([size_old_val,ids,test_scores,val_scores],f)
        
        

def get_indices_by_quantiles(s, n_sample):
    # Define the quantiles
    quantiles = np.linspace(0, 100, n_sample)

    # Initialize an empty list to store the indices
    indices = []

    # Iterate over the quantiles
    for q in quantiles:
        quantile_value = s.quantile(q / 100)  # Compute the quantile value
        indices.append(s[s <= quantile_value].sort_values(ascending=True).index[-1])  # Add the indices to the list

    return indices


def grow_data(
        model,
        X,y,
        X_test,y_test,
        file_out,
        X_val=None,y_val=None,
        n_iter: int = 50, 
        subsample: float =1,
        batch_sizes: list = None,
        grow_criterion: str = 'max_err',
        ):
    
    # record id_list in each iteration
    id_train_iter = []
    train_size = []
    train_frac = []
    test_scores={'maes':[], 'rmse':[], 'r2':[], 'maes_m2':[], 'rmse_m2':[], 'r2_m2':[]}
    val_scores={'maes':[], 'rmse':[], 'r2':[], 'maes_m2':[], 'rmse_m2':[], 'r2_m2':[]}    

    # pool ids
    pool = X.index.tolist()   
    # total number
    ntot = len(pool)

    # define the number of sammples to add in each iteration
    if batch_sizes is None:
        batch_size = int( len(pool) / n_iter)
        batch_sizes = [batch_size] * (n_iter+1)
        print(f'batch_size: {batch_size}')
        
    print(f'# entries in pool: {ntot}')
    
    # query by errors
    for n, batch_size in enumerate(batch_sizes):       
        if n == 0:       
            # initial id list
            # id_train = get_id_train_ini(y,batch_size) 
            id_train = get_indices_by_quantiles(y, batch_size)
            # # Random
            # id_train = X.sample(batch_size).index.tolist() # worth trying how this combined with batch_size changes the game            
            # another random set
            # id_train = (y - y.mean()).abs().sort_values()[-batch_size:].index.tolist()
            # id_train_iter.append(id_train)    
            
        # Update the pool by removing the samples in the training set
        pool = list(set(pool) - set(id_train))        
        n_train = len(id_train) 
        train_size.append(n_train)
        train_frac.append(n_train/ntot)

        # training
        X_train = X.loc[id_train]
        y_train = y.loc[id_train]
        X_pool = X.loc[pool]
        y_pool = y.loc[pool]

        if X_val is None:
            model.fit(X_train,y_train)
        elif X_val == 'X_pool':
            model.fit(X_train,y_train,val=(X_pool,y_pool))
        else:
            model.fit(X_train,y_train,val=(X_val,y_val))



        # test score       
        y_pred = model.predict(X_test)
        maes_test = metrics.mean_absolute_error(y_test,y_pred)
        rmse_test = metrics.mean_squared_error(y_test,y_pred,squared=False)
        r2_test = metrics.r2_score(y_test,y_pred)   
        test_scores['maes'].append(maes_test)
        test_scores['rmse'].append(rmse_test)
        test_scores['r2'].append(r2_test)
            
        
        # Get the prediction errors for the samples in pool
        y_pred = model.predict(X_pool)       

        maes_val = metrics.mean_absolute_error(y_pool,y_pred)
        rmse_val = metrics.mean_squared_error(y_pool,y_pred,squared=False)
        r2_val = metrics.r2_score(y_pool,y_pred)   
        val_scores['maes'].append(maes_val)
        val_scores['rmse'].append(rmse_val)
        val_scores['r2'].append(r2_val)

        print('')
        print(f'@ train_size: {n_train}, train_frac: {n_train/ntot:.3f}')
        print(f'@ Val scores: maes={maes_val:.3f}, rmse={rmse_val:.3f}, r2={r2_val:.3f}')
        print(f'@ Test scores: maes={maes_test:.3f}, rmse={rmse_test:.3f}, r2={r2_test:.3f}')
        print('')

        with open(file_out,'wb') as f:
            pickle.dump([train_size,train_frac,id_train_iter,test_scores,val_scores],f)

        if n_train + batch_size > ntot:
            break

        # Select the samples to add in the next round of training 
        # get the errors
        y_err = (y_pool-y_pred).abs()        
        # subsample
        y_err = y_err.sample(
            frac=subsample,
            random_state=n
            )
        y_err = y_err.sort_values(ascending=True)
        
        if grow_criterion == 'max_err':
            # get the index of the samples with the largest errors
            new_id = y_err[-batch_size:].index.tolist()
        elif grow_criterion == 'min_err':
            # get the index of the samples with the smallest errors
            new_id = y_err[:batch_size].index.tolist()

        # add the new samples to the training set
        id_train.extend(new_id)
        id_train_iter.append(new_id)

    return train_size,train_frac,id_train_iter,test_scores,val_scores
