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
|                 | Tox21\_HSE        | HSE          | Molecules  | Inherent | 423        | 257       | 10         |
| Organic Anomaly | Tox21\_MMP        | MMP          | Molecules  | Inherent | 6170       | 200       | 38         |
|                 | Tox21\_PPAR-gamma | PPAR         | Molecules  | Inherent | 219        | 252       | 15         |


### Table 2: Class-based Anomaly (TYPE II)

| Dataset Type         | Full Name      | Abbreviation | Graph Type      | OOD Type      | # ID Train | # ID Test | # OOD Test |
|----------------------|----------------|--------------|-----------------|---------------|------------|-----------|------------|
|                      | COLLAB         | -            | Social Networks | Unseen Classes | 1920       | 480       | 520        |
|                      | IMDB-BINARY    | IMDB-B       | Social Networks | Unseen Classes | 400        | 100       | 100        |
|                      | REDDIT-BINARY  | REDDIT-B     | Social Networks | Unseen Classes | 800        | 200       | 200        |
|                      | ENZYMES        | -            | Proteins        | Unseen Classes | 400        | 100       | 20         |
|                      | PROTEINS       | -            | Proteins        | Unseen Classes | 360        | 90        | 133        |
| Class-based Anomaly  | DD             | -            | Proteins        | Unseen Classes | 390        | 97        | 139        |
|                      | BZR            | -            | Molecules       | Unseen Classes | 69         | 17        | 64         |
|                      | AIDS           | -            | Molecules       | Unseen Classes | 1280       | 320       | 80         |
|                      | COX2           | -            | Molecules       | Unseen Classes | 81         | 21        | 73         |
|                      | NCI1           | -            | Molecules       | Unseen Classes | 1646       | 411       | 411        |
|                      | DHFR           | -            | Molecules       | Unseen Classes | 368        | 93        | 59         |

| Dataset Type          | Full Name                  | Abbreviation | Graph Type      | OOD Type        | # ID Train | # ID Test | # OOD Test |
|-----------------------|----------------------------|--------------|-----------------|-----------------|------------|-----------|------------|
|                       | IMDB-MULTI&rarr;IMDB-BINARY | IM&rarr;IB | Social Networks | Unseen Datasets | 1350       | 150       | 150        |
|                       | ENZYMES&rarr;PROTEINS  | EN&rarr;PR | Proteins        | Unseen Datasets | 540        | 60        | 60         |
|                       | AIDS&rarr;DHFR         | AI&rarr;DH | Molecules       | Unseen Datasets | 1800       | 200       | 200        |
|                       | BZR&rarr;COX2          | BZ&rarr;CO | Molecules       | Unseen Datasets | 364        | 41        | 41         |
|                       | ESOL&rarr;MUV          | ES&rarr;MU | Molecules       | Unseen Datasets | 1015       | 113       | 113        |
| Inter-Dataset Shift   | TOX21&rarr;SIDER       | TO&rarr;SI | Molecules       | Unseen Datasets | 7047       | 784       | 784        |
|                       | BBBP&rarr;BACE         | BB&rarr;BA | Molecules       | Unseen Datasets | 1835       | 204       | 204        |
|                       | PTC\_MR&rarr;MUTAG     | PT&rarr;MU | Molecules       | Unseen Datasets | 309        | 35        | 35         |
|                       | FREESOLV&rarr;TOXCAST  | FS&rarr;TC | Molecules       | Unseen Datasets | 577        | 65        | 65         |
|                       | CLINTOX&rarr;LIPO      | CL&rarr;LI | Molecules       | Unseen Datasets | 1329       | 148       | 148        |


### Table 4: Intra-Dataset Shift (TYPE IV)

| Dataset Type          | Full Name           | Abbreviation | Graph Type  | OOD Type        | # ID Train | # ID Test | # OOD Test |
|-----------------------|---------------------|--------------|-------------|-----------------|------------|-----------|------------|
|                       | GOOD-HIV-Size       | HIV-Size     | Molecules   | Size            | 1000       | 500       | 500        |
|                       | GOOD-ZINC-Size      | ZINC-Size    | Molecules   | Size            | 1000       | 500       | 500        |
|                       | GOOD-HIV-Scaffold   | HIV-Scaffold | Molecules   | Scaffold        | 1000       | 500       | 500        |
|                       | GOOD-ZINC-Scaffold  | ZINC-Scaffold| Molecules   | Scaffold        | 1000       | 500       | 500        |
|                       | DrugOOD-IC50-Size   | IC50-Size    | Molecules   | Size            | 1000       | 500       | 500        |
| Intra-Dataset Shift   | DrugOOD-EC50-Size   | EC50-Size    | Molecules   | Size            | 1000       | 500       | 500        |
|                       | DrugOOD-IC50-Scaffold | IC50-Scaffold | Molecules | Scaffold        | 1000       | 500       | 500        |
|                       | DrugOOD-EC50-Scaffold | EC50-Scaffold | Molecules | Scaffold        | 1000       | 500       | 500        |
|                       | DrugOOD-IC50-Assay  | IC50-Assay   | Molecules   | Protein Target  | 1000       | 500       | 500        |
|                       | DrugOOD-EC50-Assay  | EC50-Assay   | Molecules   | Protein Target  | 1000       | 500       | 500        |


