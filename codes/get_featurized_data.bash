#!/bin/bash

# get the Matminer-featurized data

for dataset in jarvis22
do 
    mkdir -p data/$dataset
    # download the featurized data from https://zenodo.org/record/8200972
    # and rename it to dat_featurized_matminer.pkl
    wget -O data/$dataset/dat_featurized_matminer.pkl https://zenodo.org/record/8200972/files/${dataset}_featurized_matminer.pkl?download=1

done

