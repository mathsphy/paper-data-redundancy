#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:51:05 2022

@author: kangming
"""
import numpy as np
import pandas as pd
# import os 
import time 
from tqdm import tqdm
# sklearn
from sklearn.model_selection import KFold, cross_validate
import tarfile
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

# For structure object conversion
import ast
from jarvis.core.atoms import Atoms
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.entries.computed_entries import ComputedStructureEntry, ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Composition

import json 
from monty.json import MontyEncoder, MontyDecoder

def strAtoms2objAtoms(str_atoms):
    """
    Convert back to a jarvis.core.atoms object from a string that was previously 
    converted from a jarvis.core.atoms object

    Parameters
    ----------
    str_atoms : str
        The string to which a jarvis.core.atoms object is converted.

    Returns
    -------
    obj_Atoms : jarvis.core.atoms

    """
    # convert the string of dict into dict: 
    # https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
    dict_atom = ast.literal_eval(str_atoms) # need module ast
    # need module jarvis
    obj_Atoms = Atoms(lattice_mat=dict_atom['lattice_mat'], 
                      coords=dict_atom['coords'], 
                      elements=dict_atom['elements'])
    
    # seems that we can simply do (https://jarvis-tools.readthedocs.io/en/master/databases.html)
    # obj_Atoms = Atoms.from_dict(str_atoms)
    
    return obj_Atoms






def strAtoms2objStructure(str_atom):
    """
    Convert strAtoms to objStructure

    Parameters
    ----------
    str_atoms : str
        The string to which a jarvis.core.atoms object is converted.

    Returns
    -------
    obj_structure : pymatgen.core.structure.Structure

    """
    
    strAtoms2objAtoms(str_atom).write_poscar() # Write POSCAR
    obj_structure = Structure.from_file('POSCAR')
    [site.to_unit_cell(in_place=True) for site in obj_structure.sites]
    return obj_structure

def dictAtoms2objStructure(dict_atom):
    """
    Convert strAtoms to objStructure

    Parameters
    ----------
    str_atoms : dict
        The dict to which a jarvis.core.atoms object is converted.

    Returns
    -------
    obj_structure : pymatgen.core.structure.Structure

    """
    obj_Atom = Atoms.from_dict(dict_atom)    
    obj_Atom.write_poscar() # Write POSCAR
    obj_structure = Structure.from_file('POSCAR')
    [site.to_unit_cell(in_place=True) for site in obj_structure.sites]
    return obj_structure

def structure2atom_dict(structure):
    from jarvis.io.vasp.inputs import Poscar       
    return Poscar.from_string(structure.to(fmt='poscar')).atoms.to_dict()

def count_sg_num(df, col='spacegroup.number'):
    '''
    Count space group occurrence in the column 'spacegroup.number' of a dataframe 

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    col : TYPE, optional
        DESCRIPTION. The default is 'spacegroup.number'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    df_sg = df[col].astype(int).value_counts().sort_index() 
    tot_sg = 230 # total number of space group
    for i in range(1,tot_sg+1):
        if i not in df_sg.index:
            df_sg[i]=0
    return df_sg.sort_index()

def dict2struct(str_struct):
    dict_struct = ast.literal_eval(str_struct) # convert sting to dict
    structure = Structure.from_dict(dict_struct)
    [site.to_unit_cell(in_place=True) for site in structure.sites]
    return structure # convert dict to Structure object

def to_unitcell(structure):
    '''
    Make sure coordinates are within the unit cell.
    Used before using structural featurizer.

    Parameters
    ----------
    structure :  pymatgen.core.structure.Structure

    Returns
    -------
    structure :  pymatgen.core.structure.Structure
    '''    
    [site.to_unit_cell(in_place=True) for site in structure.sites]
    return structure



def Featurizer(
        df,
        col_id='structure',
        ignore_errors=True,
        chunksize=16
        ):
    """
    Featurize a dataframe using Matminter featurizers

    Parameters
    ----------
    df : Pandas.DataFrame 
        DataFrame with a column named "structure"

    Returns
    -------
    A DataFrame containing labels as the first columns and features as the rest 

    """
    # For featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import (ElementProperty, 
                                                  Stoichiometry, 
                                                  ValenceOrbital, 
                                                  IonProperty)
    from matminer.featurizers.structure import (SiteStatsFingerprint, 
                                                StructuralHeterogeneity,
                                                ChemicalOrdering, 
                                                StructureComposition, 
                                                MaximumPackingEfficiency)   
    # Make sure df is a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()   
        
    # Use composition featurizers if inputs are compositions, otherwise use
    # both composition and structure featurizers
    if col_id == 'composition':
        # convert string to composition 
        a = StrToComposition()
        a._overwrite_data = True
        df[col_id] = a.featurize_dataframe(df,col_id,pbar=False)
        # no structural features
        struc_feat = []
        # 145 compositional features
        compo_feat = [
            Stoichiometry(),
            ElementProperty.from_preset("magpie"),
            ValenceOrbital(props=['frac']),
            IonProperty(fast=True)
            ]
    else:
        # Ensure sites are within unit cells
        df[col_id] = df[col_id].apply(to_unitcell)
        # 128 structural feature
        struc_feat = [
            SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"), 
            SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
            StructuralHeterogeneity(),
            MaximumPackingEfficiency(),
            ChemicalOrdering()
            ]       
        # 145 compositional features
        compo_feat = [
            StructureComposition(Stoichiometry()),
            StructureComposition(ElementProperty.from_preset("magpie")),
            StructureComposition(ValenceOrbital(props=['frac'])),
            StructureComposition(IonProperty(fast=True))
            ]
        
    # Define the featurizer
    featurizer = MultipleFeaturizer(struc_feat+compo_feat)   
    
    # Set the chunksize used for Pool.map parallelisation
    featurizer.set_chunksize(chunksize=chunksize)
    X = featurizer.featurize_dataframe(df,col_id,ignore_errors=ignore_errors)  
    
    # check failed entries    
    failed = np.any(pd.isnull(X.iloc[:,df.shape[1]:]), axis=1)
    if np.sum(failed) > 0:
        print(f'Number failed: {np.sum(failed)}/{len(failed)}')
    print('Featurization completed.')
    return X




def StructureFeaturizer(
        df_in,
        col_id='structure',
        ignore_errors=True,
        chunksize=30
        ):
    """
    Featurize a dataframe using Matminter Structure featurizer

    Parameters
    ----------
    df : Pandas.DataFrame 
        DataFrame with a column named "structure"

    Returns
    -------
    A DataFrame containing 273 features (columns)

    """
    # For featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (ElementProperty, 
                                                  Stoichiometry, 
                                                  ValenceOrbital, 
                                                  IonProperty)
    from matminer.featurizers.structure import (SiteStatsFingerprint, 
                                                StructuralHeterogeneity,
                                                ChemicalOrdering, 
                                                StructureComposition, 
                                                MaximumPackingEfficiency)
    
    
    if isinstance(df_in, pd.Series):
        df = df_in.to_frame()
    else:
        df = df_in
    df[col_id] = df[col_id].apply(to_unitcell)
    
    # 128 structural feature
    struc_feat = [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"), 
        SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
        StructuralHeterogeneity(),
        MaximumPackingEfficiency(),
        ChemicalOrdering()
        ]       
    # 145 compositional features
    compo_feat = [
        StructureComposition(Stoichiometry()),
        StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=['frac'])),
        StructureComposition(IonProperty(fast=True))
        ]
    featurizer = MultipleFeaturizer(struc_feat+compo_feat)    
    # Set the chunksize used for Pool.map parallelisation
    featurizer.set_chunksize(chunksize=chunksize)
    featurizer.fit(df[col_id])
    X = featurizer.featurize_dataframe(df=df,col_id=col_id,ignore_errors=ignore_errors)  
    # check failed entries    
    print('Featurization completed.')
    failed = np.any(pd.isnull(X.iloc[:,df.shape[1]:]), axis=1)
    if np.sum(failed) > 0:
        print(f'Number failed: {np.sum(failed)}/{len(failed)}')
    return X


def get_cv_scores(model, X, Y, n_splits=5, shuffle=True):
    """
    Get the CV scores

    Parameters
    ----------
    model : TYPE
    X : df of features
    Y : df of the target
    n_splits = 10
    shuffle = True

    Returns
    -------
    scores

    """    
    start = time.time()
    
    cv = KFold(n_splits=n_splits, shuffle=shuffle)
    
    scoring = {'rmse': 'neg_root_mean_squared_error',
               'maes': 'neg_mean_absolute_error',
               'mape': 'neg_mean_absolute_percentage_error'}        
    if n_splits != X.shape[0]:
        scoring.update({'r2': 'r2'})  

    scores = cross_validate(model, X, Y, scoring=scoring,
                            return_train_score=True,
                            cv=cv, n_jobs=-1)
    # Metrics for train set
    rmse = np.mean(np.abs(scores["train_rmse"]))
    maes = np.mean(np.abs(scores["train_maes"]))
    mape = np.mean(np.abs(scores["train_mape"]))
    if n_splits != X.shape[0]:
        r2 = np.mean(np.abs(scores["train_r2"]))
    else:
        r2 = 0
    print(f'Train set: \n rmse = {rmse:.3f}, maes = {maes:.3f}, mape = {mape: .3f}, r2={r2:.4f}')
    # Metrics for test set
    rmse = np.mean(np.abs(scores["test_rmse"]))
    maes = np.mean(np.abs(scores["test_maes"]))
    mape = np.mean(np.abs(scores["test_mape"]))
    if n_splits != X.shape[0]:
        r2 = np.mean(np.abs(scores["test_r2"]))
    else:
        r2 = 0
    print(f'Test set: \n rmse = {rmse:.3f}, maes = {maes:.3f}, mape = {mape: .3f}, r2={r2:.4f}')
    # Get elasped time
    end = time.time()
    dtime = end-start
    if dtime < 60:
        print(f'Time: {dtime: .1f} secs')
    else:
        print(f'Time: {round(dtime/60,1): .1f} mins')
        
    return scores



def MeasureInfluence_old(model, X, Y, log_trans = True):
    """
    Influence measure based on deletion diagnostics 

    Parameters
    ----------
    model : model to fit
    X : features
    Y : target
    Returns
    -------
    Influence measure

    """ 
    if log_trans == False:
        model.fit(X,Y)
        Y_pred = model.predict(X)
    else:
        model.fit(X, np.log10(Y))
        Y_pred = 10 ** model.predict(X)
    
    abs_influ = []
    pct_influ = []
    nsample = X.shape[0]

    for i in range(nsample):
        X_drop = X.drop(X.index[i])
        Y_drop = Y.drop(Y.index[i])
        if log_trans == False:
            model.fit(X_drop,Y_drop)
            Y_pred_by_drop = model.predict(X)
        else:
            model.fit(X_drop,np.log10(Y_drop))
            Y_pred_by_drop = 10 ** model.predict(X)
        abs_influ.append(np.abs(Y_pred_by_drop - Y_pred))
        pct_influ.append(np.abs((Y_pred_by_drop - Y_pred)/Y_pred))
    
    abs_influ = pd.DataFrame(abs_influ, index=X.index, columns=X.index)
    pct_influ = pd.DataFrame(pct_influ, index=X.index, columns=X.index)
    return abs_influ, pct_influ


def MeasureInfluence(model, X, Y, log_trans = True, rm_self = False):
    """
    Influence measure based on deletion diagnostics 

    Parameters
    ----------
    model : 
        model to fit
    X : pd.DataFrame
        Feature
    Y : pd.Series
        Target
    log_trans :
        whether perform log transformation before training
    rm_self:
        whether remove j when 
        
    Returns
    -------
    Influence measure

    """ 
    # Function to train and predict
    def pred_log_trans(X, Y, df): 
        if isinstance(df, pd.DataFrame):
            index = df.index
            X_test = df.to_numpy()
        else: # Series 
            index = None
            X_test = df.to_numpy().reshape((1,-1))
        
        if log_trans == False:
            model.fit(X.to_numpy(),Y.to_numpy())
            Y_test = model.predict(X_test)
        else:
            model.fit(X.to_numpy(), np.log10(Y))
            Y_test = 10 ** model.predict(X_test)
            
        if Y_test.size == 1:
            Y_test = Y_test.item() # return scalar
        else:
            Y_test = pd.Series(Y_test, index=index)
        return Y_test           
        
    abs_influ = []
    pct_influ = []
    nsample = X.shape[0]
    index=X.index
    # Case 1 (remove only i): 
    # Look at the influence of removing i on j
    if rm_self == False:
        # Prediction used as the reference
        Y_ref = pred_log_trans(X, Y, X)
        for i in range(nsample):
            X_dropi = X.drop(index[i])
            Y_dropi = Y.drop(index[i])
            # prediction used to measure influence
            Y_influ = pred_log_trans(X_dropi, Y_dropi, X)           
            abs_influ.append(np.abs(Y_influ - Y_ref))
            pct_influ.append(np.abs((Y_influ - Y_ref)/Y_ref))
    # Case 2 (remove both i and j)
    else:
        for i in tqdm(range(nsample)):
            X_dropi = X.drop(index[i])
            Y_dropi = Y.drop(index[i])    
            # Prediction used as the reference
            Y_ref = pred_log_trans(X_dropi, Y_dropi, X)
            # prediction used to measure influence
            Y_influ = []
            for j in range(nsample):
                if i == j:
                    y_j = Y_ref[index[i]]
                else:                    
                    X_dropij = X_dropi.drop(index[j])
                    Y_dropij = Y_dropi.drop(index[j]) 
                    x_j = X.loc[index[j], :]
                    y_j = pred_log_trans(X_dropij, Y_dropij, x_j)
                    
                Y_influ.append(y_j)
            # Convert to Series
            Y_influ = pd.Series(Y_influ, index=index)
            abs_influ.append(np.abs(Y_influ - Y_ref))
            pct_influ.append(np.abs((Y_influ - Y_ref)/Y_ref))
            
    abs_influ = pd.DataFrame(abs_influ, index=index, columns=index)
    pct_influ = pd.DataFrame(pct_influ, index=index, columns=index)
    return abs_influ, pct_influ






def calc_Ef(dat_all, reset_reference):
    
    from collections import Counter
    
    def set_reference_energy(df,reset_reference):
        if reset_reference:            
            tmp = dat_all[dat_all['nelements']==1]
            dict_pure={}
            for i in tmp.index:
                element=tmp.loc[i,'elements'][0]
                e_per_atom=tmp.loc[i,'e_per_atom']
                dict_pure[element] = e_per_atom
        else:    
            # Total energy per atom (my setttings using settings relaxed)
            dict_pure = {
                # 'Mo': -10.90971728,
                'Al': -3.74798896,
                'Fe': -8.23919192,
                'Si': -5.425396365,
                'Ni': -5.46770387,
                'Cu': -3.73300732,
                'Mn': -8.97873597827586,
                'Co': -7.036824315,
                # 'Ti': -7.8400272766666665,
                'Cr': -9.48494376
                }
        return dict_pure
    # # Total energy per atom (my settings using MP final structure)
    # dict_pure = {'Al': -3.74799,
    #              'Fe': -8.2384,
    #              'Si': -5.42539,
    #              'Ni': -5.4671,
    #              'Mn': -8.98245,
    #              'Co': -7.03653,
    #              'Cr': -9.49614}      
    dict_pure = set_reference_energy(dat_all,
                                     reset_reference=reset_reference)
    
    
    # print(dict_pure)    
    # list_E2subtract = []
    list_Ef = []
    for row in dat_all.index:        
        e_per_atom = dat_all.loc[row, 'e_per_atom']
        structure = dat_all.loc[row, 'structure'] 
        elements = [str(i) for i in structure.species] 
        count_element = Counter(elements)    
        E2subtract = 0
        NIONS = 0
        for element, natoms in count_element.items():
            E2subtract += dict_pure[element] * natoms
            NIONS += natoms
        Ef = e_per_atom - E2subtract/NIONS
        list_Ef.append(Ef)    
        
    dat_Ef = pd.Series(list_Ef)
    return dat_Ef









def read_outdir(outdir, include_structure_ini, reset_reference):
    
    
    dat_all = pd.read_csv(f'{outdir}/outfile')   
    
    list_structure = []
    list_structure_ini=[]
    list_elements = []
    list_nelements = []
    for index, fid in dat_all['fid'].items():
        structure = Structure.from_file(f'{outdir}/POSCAR.{fid}')
        # put sites into unit cell (to_unit_cell=True)
        # because structure featurizer fails when fractional coord > 1
        [site.to_unit_cell(in_place=True) for site in structure.sites]
        try:
            structure_ini = Structure.from_file(f'{outdir}/initial_structures/POSCAR.{fid}')
            [site.to_unit_cell(in_place=True) for site in structure_ini.sites]
        except:
            structure_ini = None            
        elements = [str(i) for i in structure.species]     
        list_structure.append(structure)
        list_structure_ini.append(structure_ini)
        elements = sorted(list(set(elements)))
        list_elements.append(elements)
        list_nelements.append(len(elements))

    if include_structure_ini:
        dat_all['structure_ini'] = pd.Series(list_structure_ini)   
        dat_all['structure_ini_as_dict'] = dat_all['structure_ini'].apply(lambda x: x.as_dict())    
        space_group_number_ini = dat_all['structure_ini'].apply(lambda x: SpacegroupAnalyzer(x).get_space_group_number())
    else:
        dat_all['structure_ini'] = None
        dat_all['structure_ini_as_dict'] = None
        space_group_number_ini = None        
    dat_all.insert(1, 'space_group_number_ini', space_group_number_ini) 

    dat_all['structure'] = pd.Series(list_structure)
    dat_all['structure_as_dict'] = dat_all['structure'].apply(lambda x: x.as_dict())
    space_group_number = dat_all['structure'].apply(lambda x: SpacegroupAnalyzer(x).get_space_group_number())
    dat_all.insert(1, 'space_group_number', space_group_number)     
    dat_all['elements'] = pd.Series(list_elements)
    dat_all['nelements'] = pd.Series(list_nelements)
    dat_all['e_per_atom'] = dat_all['energy']/dat_all['NIONS']  
    dat_all['volume_per_atom'] = dat_all['structure'].apply(lambda x: x.volume)/dat_all['NIONS']
    dat_all['Ef_per_atom'] = calc_Ef(dat_all, reset_reference=reset_reference)    
    dat_all = dat_all.set_index('fid')
    return dat_all


# def read_targz(filename, include_structure_ini, reset_reference):
    
#     tar = tarfile.open(filename,"r:gz")
#     outfile = tar.extractfile('outdir/outfile')
#     dat_all = pd.read_csv(outfile)
        
#     list_structure = []
#     list_structure_ini=[]
#     list_elements = []
#     list_nelements = []
#     for index, fid in dat_all['fid'].items():
#         poscar = tar.extractfile(f'outdir/POSCAR.{fid}')
#         structure = Structure.from_file(poscar)
#         # put sites into unit cell (to_unit_cell=True)
#         # because structure featurizer fails when fractional coord > 1
#         [site.to_unit_cell(in_place=True) for site in structure.sites]
#         try:
#             poscar_ini = tar.extractfile(f'outdir/initial_structures/POSCAR.{fid}')
#             structure_ini = Structure.from_file(poscar_ini)
#             [site.to_unit_cell(in_place=True) for site in structure_ini.sites]
#         except:
#             structure_ini = None            
#         elements = [str(i) for i in structure.species]     
#         list_structure.append(structure)
#         list_structure_ini.append(structure_ini)
#         elements = sorted(list(set(elements)))
#         list_elements.append(elements)
#         list_nelements.append(len(elements))

#     if include_structure_ini:
#         dat_all['structure_ini'] = pd.Series(list_structure_ini)   
#         dat_all['structure_ini_as_dict'] = dat_all['structure'].apply(lambda x: x.as_dict())    
#         space_group_number_ini = dat_all['structure_ini'].apply(lambda x: SpacegroupAnalyzer(x).get_space_group_number())
#         dat_all.insert(1, 'space_group_number_ini', space_group_number_ini) 

#     dat_all['structure'] = pd.Series(list_structure)
#     dat_all['structure_as_dict'] = dat_all['structure'].apply(lambda x: x.as_dict())
#     space_group_number = dat_all['structure'].apply(lambda x: SpacegroupAnalyzer(x).get_space_group_number())
#     dat_all.insert(1, 'space_group_number', space_group_number)     
#     dat_all['elements'] = pd.Series(list_elements)
#     dat_all['nelements'] = pd.Series(list_nelements)
#     dat_all['e_per_atom'] = dat_all['energy']/dat_all['NIONS']  
#     dat_all['Ef_per_atom'] = calc_Ef(dat_all, reset_reference=reset_reference)    
#     dat_all = dat_all.set_index('fid')
#     return dat_all





def flatten_list(xss):
    ''' Flatten list of lists into a list '''
    return [x for xs in xss for x in xs]

def gen_sys_list(elements):
    max_nelements = len(elements)
    # Reorder, necessary for the query ('Fe-Ni' can be recognized by 'Ni-Fe' cannot)
    element_list = sorted(elements)

    # unary systems
    sys_list_una = element_list
    # binary systems
    sys_list_bin = []
    # tenary systems
    sys_list_ter = []
    # quaternary systems
    sys_list_qua = []
    # quinary systems
    sys_list_qui = []

    # create the query list
    for i in range(max_nelements):
        for j in range(i+1,max_nelements):
            sys_list_bin.append( element_list[i]+"-"+element_list[j] )
            for k in range(j+1,max_nelements):
                sys_list_ter.append(element_list[i]+"-"+element_list[j]+"-"+element_list[k])
                for l in range(k+1,max_nelements):
                    sys_list_qua.append(element_list[i] + "-" + element_list[j] + "-" + element_list[k] + "-" + element_list[l])
                    for m in range(l+1,max_nelements):
                        #sys_list_qui.append(element_list[i] + "-" + element_list[j] + "-" + element_list[k] + "-" + element_list[l]+ "-" + element_list[m])
                        sys_list_qui.append('-'.join([element_list[id] for id in [i,j,k,l,m]]))
                        
    print('Possible combinaisons:')
    print('unary: {}'.format(len(sys_list_una)))
    print('binary: {}'.format(len(sys_list_bin)))
    print('ternary: {}'.format(len(sys_list_ter)))
    print('quaternary: {}'.format(len(sys_list_qua)))
    print('quinary: {}'.format(len(sys_list_qui))) # only one entry
    sys_list = [sys_list_una, sys_list_bin, sys_list_ter, sys_list_qua,sys_list_qui]
    return flatten_list(sys_list)


def get_phase_diagram(dat_all,dict_pure=None,use_total_e=True):
    if use_total_e:
        if 'energy' not in dat_all.columns:
            dat_all['energy'] = dat_all['NIONS'] * dat_all['e_per_atom']
    else:
        dat_all['energy'] = dat_all['NIONS'] * dat_all['Ef_per_atom']            
    # Entries of compounds
    entries = []
    for idx in dat_all.index:
        entry = ComputedEntry(composition = dat_all.loc[idx,'formula'],
                              energy = dat_all.loc[idx,'energy'],
                              entry_id = idx)
        entries.append(entry)
    dat_all['entry'] = pd.Series(entries,index=dat_all.index)  
    if use_total_e:
        if dict_pure is None:
            # print('Error')
            # Entries of terminal elements
            # dict_pure = {'Al': -3.75298504,
            #              'Fe': -8.23731135,
            #              'Si': -5.425394935,
            #              'Ni': -5.46824999,
            #              'Mn': -8.99325391137931,
            #              'Co': -7.03683039,
            #              'Cr': -9.49894925}
            dict_pure = {
                # 'Mo': -10.90971728,
                'Al': -3.74798896,
                'Fe': -8.23919192,
                'Si': -5.425396365,
                'Ni': -5.46770387,
                'Cu': -3.73300732,
                'Mn': -8.97873597827586,
                'Co': -7.036824315,
                # 'Ti': -7.8400272766666665,
                'Cr': -9.48494376
                }
    else:
        # Entries of terminal elements
        dict_pure = {'Al': 0,
                     'Fe': 0,
                     'Si': 0,
                     'Ni': 0,
                     'Mn': 0,
                     'Co': 0,
                     'Cr': 0,
                     'Cu': 0
                     }      
    entries_pure=[]
    for idx, energy in dict_pure.items():
        entry = ComputedEntry(composition=idx,
                              energy=energy)
        entries_pure.append(entry)
    #
    phase_diagram = PhaseDiagram(entries + entries_pure)
    
    if use_total_e:
        json_pd = 'phase_diagram_calculated_with_e_total.json'
    else:
        json_pd = 'phase_diagram_calculated_with_e_form.json'
        
    with open(json_pd, 'w') as f:
        json.dump(phase_diagram, f, cls=MontyEncoder)
    
    return phase_diagram
    

# def get_entry_and_e_above_hull(phase_diagram,dat_all,tols=[1E-8,1E-9,1E-10]):
#     dat_all['e_above_hull'] = dat_all['entry'].apply(
#         lambda x: phase_diagram.get_e_above_hull(x)    
#         )
#     dat_all['e_decomposed'] = dat_all['entry'].apply(
#         lambda x: phase_diagram.get_phase_separation_energy(x,tols)    
#         )    
#     return dat_all







# def update_phase_diagram(phase_diagram, df, col_energy,):
#     for idx in df.index:
#         entry = ComputedEntry(composition = df.loc[idx,'formula'],
#                               energy = df.loc[idx,'energy'],
#                               entry_id = idx)






def get_transition_T(composition_str: str, e_per_atom: float, phase_diagram):
    compo_obj = Composition(composition_str)
    compo_obj = compo_obj.fractional_composition
    entry = ComputedEntry(composition=compo_obj,
                          energy=e_per_atom)
    e_hull = phase_diagram.get_e_above_hull(entry)    
    kB = 8.617333262 * 10**(-5)
    S_conf = 0
    for c in compo_obj.as_dict().values():
        S_conf += ( -c * math.log(c) )    
    transition_T = e_hull/(kB*S_conf)
    return transition_T
    
    
    # ComputedEntry(composition = dat_all.loc[idx,'formula'],
    #                       energy = dat_all.loc[idx,'energy'],
    #                       entry_id = idx )



# def rank_corr(df, cutoff=0.7):
#     corr = df.corr().abs()
    





def custom_matplotlib():
    import matplotlib as mpl
    
    ''' Figure '''    
    mpl.rcParams['figure.figsize'] = [4.5,4.5]
    mpl.rcParams['figure.dpi'] = 150    
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['legend.framealpha'] = 1
    '''
    Font: 
        https://www.statology.org/change-font-size-matplotlib/#:~:text=Note%3A%20The%20default%20font%20size%20for%20all%20elements%20is%2010.
        https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

    '''
    font = {
        #'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 14}
    mpl.rc('font', **font)
    



'''
For distilaltion
'''


def drop_failed_structures(df,label_stuc=None):
    # By default
    if label_stuc is None:
        label_stuc = df.columns[-273:-145]
    # Drop structures without structural features 
    df_failed_struct = df[df[label_stuc].isnull().any(axis=1)]
    num_tot = df.shape[0]
    print(f'Total: {num_tot}')
    num_failed = df_failed_struct.shape[0]
    print(f'Failed structural featurization: {num_failed}')
    num_sucess = num_tot - num_failed
    print(f'Sucess: {num_sucess}')    
    return df.drop(df_failed_struct.index)


def get_col2drop(X, cutoff=0.6, method='pearson',by=['Count', 'Max', 'Mean']): 
    # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    # Documentation for the get_corr_mat function
    '''
    return a list of columns to drop and to keep based on the cutoff value
    '''

    corr_all = X.corr(method=method).abs()   
    np.fill_diagonal(corr_all.values, 0)
    col2drop=[]
    for n in range(corr_all.shape[0]):
        # Current correlation matrix
        corr = corr_all.drop(columns=col2drop,index=col2drop)    
        corr_max = corr.max().rename('Max')
        corr_mean = corr.mean().rename('Mean')
        count = (corr >= cutoff).sum().rename('Count')    
        count = pd.concat([count,corr_max,corr_mean],axis=1).sort_values(
            by=by
            )    
        print(count.tail())  
        if count['Count'][-1] > 0:
            newcol2drop = count.index[-1]
            col2drop.append(newcol2drop)
            # print(f'{len(col2drop)} features to drop.')
        else:
            col2keep = list(set(X.columns.tolist()) - set(col2drop))
            print(f'{len(col2drop)} features to drop')
            print(f'{len(col2keep)} features to keep')
            return  col2keep, col2drop



def get_scores(model,X_train,y_train,X_test,y_test, X_val=None, y_val=None):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    maes = metrics.mean_absolute_error(y_test,y_pred)
    rmse = metrics.mean_squared_error(y_test,y_pred,squared=False)
    r2 = metrics.r2_score(y_test,y_pred)
    print(f'Test scores: MAE={maes:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}')
    if (X_val is not None) and (y_val is not None):
        y_pred = model.predict(X_val)
        maes_val = metrics.mean_absolute_error(y_val,y_pred)
        rmse_val = metrics.mean_squared_error(y_val,y_pred,squared=False)
        r2_val = metrics.r2_score(y_val,y_pred)
        print(f'Val scores: MAE={maes_val:.3f}, RMSE={rmse_val:.3f}, R2={r2_val:.3f}')
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')
        return maes, rmse, r2, maes_val, rmse_val, r2_val
    else:
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')
        return maes, rmse, r2
    
#%%
'''
Should make something that:
    take a fraction to fit, but transform on the whole
'''

def do_umap(fsave,X,n_neighbors,metric,precomputed_knn=(None,None,None),
            rdsave=None,reducer=None,densmap=False,verbose=True
            ):    
    import umap
    import pickle
    '''
    Use UMAP to fit and/or transform.

    Parameters
    ----------
    fsave : csv filename to save the UMAP embeddings 
    X : input feature df
    n_neighbors : n_neighbors
    metric : distance metric
    precomputed_knn : precomputed_knn, used to save time. 
    rdsave : pickle filename to save the fitted UAMP object. 
    reducer : The fitted UAMP object. Used to transform the new data without
        refitting the UAMP object

    Returns
    -------
    z_umap : the UMAP embeddings 
    reducer : the fitted UAMP object

    '''
    if reducer is None:
        print('No reducer provided, constructing one...')
        reducer = umap.UMAP(
            # random_state=random_state, # setting random_state will disable multithreading
            # low_memory = False,
            n_components=2,
            n_neighbors=n_neighbors,
            precomputed_knn=precomputed_knn,
            densmap=densmap,
            verbose=verbose
            )     
        z_umap = pd.DataFrame(
            reducer.fit_transform(X),index = X.index, columns=[0,1]
            )
        if rdsave is not None:
            try:
                pickle.dump(reducer,open(rdsave,'wb'))
            except Exception as err:
                print(err)
    else:
        print('Reducer provided, transforming ...')
        z_umap = pd.DataFrame(
            reducer.transform(X),index = X.index, columns=[0,1]
            )         
    z_umap.to_csv(fsave)
    print(f'UMAP for n_neighbors={n_neighbors} finished')
    return z_umap, reducer
