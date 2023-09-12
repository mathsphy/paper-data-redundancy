# paper-data-redundancy
---
Codes and data for the paper ["On the redundancy in large material datasets"](https://arxiv.org/abs/2304.13076) by 
Kangming Li, 
Daniel Persaud, 
Kamal Choudhary, 
Brian DeCost, 
Michael Greenwood, 
Jason Hattrick-Simpers. The abstract of the paper is as follows:
> Extensive efforts to gather materials data have largely overlooked potential data redundancy. In this study, we present evidence of a significant degree of redundancy across multiple large datasets for various material properties, by revealing that up to 95 % of data can be safely removed from machine learning training with little impact on in-distribution prediction performance. The redundant data is related to over-represented material types and does not mitigate the severe performance degradation on out-of-distribution samples. In addition, we show that uncertainty-based active learning algorithms can construct much smaller but equally informative datasets. We discuss the effectiveness of informative data in improving prediction performance and robustness and provide insights into efficient data acquisition and machine learning training. This work challenges the "bigger is better" mentality and calls for attention to the information richness of materials data rather than a narrow emphasis on data volume.

## Dependencies
Please follow `setup_env.bash` to setup the python environment.
```
env_name="test"
conda create -n $env_name -y python=3.10
conda activate $env_name
conda install -y -c conda-forge scikit-learn py-xgboost-gpu  pandas matplotlib

# For data preprocessing and featurization 
conda install -y -c conda-forge pymatgen

# For getting uncertainty estimates of XGBoost
pip install ibug

# For using ALIGNN
pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install alignn 
```



## Data
The JARVIS, Materials Project and OQMD dataset snapshots considered in this study and their description can be found on [Zenodo](https://zenodo.org/record/8200972).

## Demo
***Under construction.***

The bash files `get_featurized_data.bash` and `run_al.bash` are provided to reproduce the QBC active learning results for the JARVIS22 dataset.
