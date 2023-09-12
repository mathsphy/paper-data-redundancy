#!/bin/bash

env_name="test"
conda create -n $env_name -y python=3.10
conda activate $env_name

conda install -y -c conda-forge scikit-learn py-xgboost-gpu  pandas matplotlib

# For data preprocessing and featurization 
conda install -y -c conda-forge pymatgen

# For getting uncertainty estimates of XGBoost
pip install ibug

# # For using ALIGNN
# pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html
# pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
# pip install alignn 

