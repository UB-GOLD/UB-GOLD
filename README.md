# (UB-GOLD) Unifying Graph-Level Out-of-Distribution Detection and Anomaly Detection: A Benchmark

This is the official implementation of the following paper:

> [ Unifying Graph-Level Out-of-Distribution Detection and Anomaly Detection: A Benchmark](https://arxiv.org/abs/2306.122)
> 
> Submitted to NeurIPS 2024 Datasets and Benchmarks Track

Environment Setup
-----------------

Before you begin, ensure that you have Anaconda or Miniconda installed on your system. This guide assumes that you have a CUDA-enabled GPU.

**Required Dependencies** :

* torch>=2.0.1
* torch_geometric>=2.4.0
* python>=3.8
* numpy>=1.20.3
* scikit-learn>=1.0.2
* scipy>=1.8.0
* networkx>=2.6.3
* rdkit>=2023.3.1
* dgl>=2.1.0
* pygcl>=0.1.2
```shell
# Create and activate a new Conda environment named 'UBGOLD'
conda create -n UBGOLD
conda activate UBGOLD

# Install Pytorch and Pyg with CUDA 11.8 support
# If your use a different CUDA version, please refer to the PyTorch and Pyg websites for the appropriate versions.

pip install -r requirements.txt

cd ./UBGOLD

python benchmark/mymain.py
```
### Table 1: Organic Anomaly (TYPE I)

| Dataset Type    | Full Name         | Abbreviation | Graph Type | OOD Type | # ID Train | # ID Test | # OOD Test |
|-----------------|-------------------|--------------|------------|----------|------------|-----------|------------|
|                 | Tox21\_p53        | p53          | Molecules  | Inherent | 8088       | 241       | 28         |
| \colorbox{red!10}{(TYPE I)}         | Tox21\_HSE        | HSE          | Molecules  | Inherent | 423        | 257       | 10         |
| Organic Anomaly | Tox21\_MMP        | MMP          | Molecules  | Inherent | 6170       | 200       | 38         |
|                 | Tox21\_PPAR-gamma | PPAR         | Molecules  | Inherent | 219        | 252       | 15         |

