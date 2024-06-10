<div align="center">
  <img src="https://github.com/UB-GOLD/UB-GOLD/blob/main/Image/Show.jpg" alt="UB-GOLD" style="margin-bottom: -20px;" width="600px"/>
</div>


# UB-GOLD: Unified Benchmark for unsupervised Graph-level OOD and anomaLy Detection

This is the official implementation of the following paper:

> [Unifying Unsupervised Graph-Level Out-of-Distribution Detection and Anomaly Detection: A Benchmark](https://arxiv.org/abs/2306.122)
> 
> Submitted to NeurIPS 2024 Datasets and Benchmarks Track

Environment Setup
-----------------

Before you begin, ensure that you have Anaconda or Miniconda installed on your system. This guide assumes that you have a CUDA-enabled GPU.

**Required Dependencies** :

* torch>=2.0.1
* torch_geometric>=2.4.0
* python>=3.8
* scikit-learn>=1.0.2
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
```

Codebase Folder
-----------------
```
├── ...
├── benchmark (Benchmark execution)
│   ├── Source (Default hyperparameters)
│   ├── my_random_search.py
│   ├── mymain.py
│   ├── ...
├── results (Save the results)
├── data (Raw datasets)
├── GAOOD (Methods)
│   ├── nn
│   ├── detector
│   ├── ...
├── dataloader (Data preprocessing pipeline)
│   ├── ...
├── ...
```

Dataset Preparation
-----------------
The data used in UB-GOLD should be downloaded in './data', and they can be downloaded from the following sources:
1. [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/)
    - COLLAB, IMDB-BINARY, REDDIT-BINARY, ENZYMES, PROTEINS
    - DD, BZR, AIDS, COX2, NCI1, DHFR
2. [OGB](https://github.com/snap-stanford/ogb) (It can be downloaded automatically)
   - BBBP, BACE, CLINTOX, LIPO, FREESOLV
   - TOXCAST, SOL, MUV, TOX21,SIDER
3. [TOX21](https://tripod.nih.gov/tox21/challenge/data.jsp#)
   - Tox21\_p53, Tox21\_HSE, Tox21\_MMP,Tox21\_PPAR-gamma
4. [DrugOOD](https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC)
   - IC50 (SIZE,SCAFFOLD,ASSAY)
   - EC50 (SIZE,SCAFFOLD,ASSAY)
5. GOOD
   - [GOODHIV](https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view)
   - [GOODZINC](https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view)
     
Benchmarking
-----------------

**With Default Hyperparameters**
```
cd ./UBGOLD
```
The main experiments:
```
python benchmark/mymain.py 

OR

cp ./benchmark/Source/GOOD-D.sh .

bash GOOD-D.sh
```
The far-near OOD detection experiments:

```
python benchmark/near_far_ood.py 

OR

cp ./benchmark/Source/near_far_ood.sh .

bash near_far_ood.sh
```

The perturbation OOD detection experiments:

```
python benchmark/per_ood.py 

OR

cp ./benchmark/Source/per_ood.sh .

bash per_ood.sh
```

**With Optimal Hyperparameters through Random Search**
```
python benchmark/my_random_search.py

OR

cp ./benchmark/Source/search.sh .

bash search.sh
```



Supported Methods  (16 Methods)
-----------------
This part lists all the methods we include in this codebase. We support 16 popular methods for anomaly detection and OOD detection.
### Table 1: 2-Step Methods

| Category | Models and References |
|----------|------------------------|
|Graph kernel with detector | [PK-SVM](https://arxiv.org/pdf/2012.12931.pdf), [PK-IF](https://arxiv.org/pdf/2012.12931.pdf), [WL-SVM](https://arxiv.org/pdf/2012.12931.pdf), [WL-IF](https://arxiv.org/pdf/2012.12931.pdf)  |
|Self-supervised learning with detector  | [IG-SVM](https://arxiv.org/pdf/2211.04208.pdf), [IG-IF](https://arxiv.org/pdf/2211.04208.pdf), [GCL-SVM](https://arxiv.org/pdf/2211.04208.pdf), [GCL-IF](https://arxiv.org/pdf/2211.04208.pdf)  |

### Table 2: End-to-End Methods

| Category | Models and References |
|----------|------------------------|
| Graph neural network-based GLAD |[OCGIN](https://arxiv.org/pdf/2012.12931.pdf), [GLADC](https://www.nature.com/articles/s41598-022-22086-3), [GLocalKD](https://arxiv.org/pdf/2112.10063.pdf), [OCGTL](https://arxiv.org/pdf/2205.13845.pdf), [SIGNET](https://proceedings.neurips.cc/paper_files/paper/2023/file/1c6f06863df46de009a7a41b41c95cad-Paper-Conference.pdf), [CVTGAD](https://link.springer.com/chapter/10.1007/978-3-031-43412-9_11)    |
| Graph neural network-based GLOD  | [GOOD-D](https://arxiv.org/pdf/2211.04208.pdf) , [GraphDE](https://proceedings.neurips.cc/paper_files/paper/2022/file/c34262c35aa5f8c1a091822cbb2020c2-Paper-Conference.pdf)  |



Supported 4 TYPE Datasets (35 Datasets)
-----------------
This part lists all the datasets we include in our Benchmark.

### Table 1: Organic Anomaly (TYPE I)

| Full Name         | Graph Type | OOD Type | # ID Train | # ID Test | # OOD Test |
|-------------------|------------|----------|------------|-----------|------------|
| Tox21\_p53        | Molecules  | Inherent | 8088       | 241       | 28         |
| Tox21\_HSE        | Molecules  | Inherent | 423        | 257       | 10         |
| Tox21\_MMP        | Molecules  | Inherent | 6170       | 200       | 38         |
| Tox21\_PPAR-gamma | Molecules  | Inherent | 219        | 252       | 15         |

### Table 2: Class-based Anomaly (TYPE II)

| Full Name      | Graph Type      | OOD Type | # ID Train | # ID Test | # OOD Test |
|----------------|-----------------|----------|------------|-----------|------------|
| COLLAB         | Social Networks | UC       | 1920       | 480       | 520        |
| IMDB-BINARY    | Social Networks | UC       | 400        | 100       | 100        |
| REDDIT-BINARY  | Social Networks | UC       | 800        | 200       | 200        |
| ENZYMES        | Proteins        | UC       | 400        | 100       | 20         |
| PROTEINS       | Proteins        | UC       | 360        | 90        | 133        |
| DD             | Proteins        | UC       | 390        | 97        | 139        |
| BZR            | Molecules       | UC       | 69         | 17        | 64         |
| AIDS           | Molecules       | UC       | 1280       | 320       | 80         |
| COX2           | Molecules       | UC       | 81         | 21        | 73         |
| NCI1           | Molecules       | UC       | 1646       | 411       | 411        |
| DHFR           | Molecules       | UC       | 368        | 93        | 59         |

### Table 3: Inter-Dataset Shift (TYPE III)

| Full Name                  | Graph Type      | OOD Type | # ID Train | # ID Test | # OOD Test |
|----------------------------|-----------------|----------|------------|-----------|------------|
| IMDB-MULTI&rarr;IMDB-BINARY | Social Networks | UD       | 1350       | 150       | 150        |
| ENZYMES&rarr;PROTEINS       | Proteins        | UD       | 540        | 60        | 60         |
| AIDS&rarr;DHFR              | Molecules       | UD       | 1800       | 200       | 200        |
| BZR&rarr;COX2               | Molecules       | UD       | 364        | 41        | 41         |
| ESOL&rarr;MUV               | Molecules       | UD       | 1015       | 113       | 113        |
| TOX21&rarr;SIDER            | Molecules       | UD       | 7047       | 784       | 784        |
| BBBP&rarr;BACE              | Molecules       | UD       | 1835       | 204       | 204        |
| PTC\_MR&rarr;MUTAG          | Molecules       | UD       | 309        | 35        | 35         |
| FREESOLV&rarr;TOXCAST       | Molecules       | UD       | 577        | 65        | 65         |
| CLINTOX&rarr;LIPO           | Molecules       | UD       | 1329       | 148       | 148        |

### Table 4: Intra-Dataset Shift (TYPE IV)

| Full Name           | Graph Type  | OOD Type       | # ID Train | # ID Test | # OOD Test |
|---------------------|-------------|----------------|------------|-----------|------------|
| GOOD-HIV-Size       | Molecules   | Size           | 1000       | 500       | 500        |
| GOOD-ZINC-Size      | Molecules   | Size           | 1000       | 500       | 500        |
| GOOD-HIV-Scaffold   | Molecules   | Scaffold       | 1000       | 500       | 500        |
| GOOD-ZINC-Scaffold  | Molecules   | Scaffold       | 1000       | 500       | 500        |
| DrugOOD-IC50-Size   | Molecules   | Size           | 1000       | 500       | 500        |
| DrugOOD-EC50-Size   | Molecules   | Size           | 1000       | 500       | 500        |
| DrugOOD-IC50-Scaffold | Molecules | Scaffold       | 1000       | 500       | 500        |
| DrugOOD-EC50-Scaffold | Molecules | Scaffold       | 1000       | 500       | 500        |
| DrugOOD-IC50-Assay  | Molecules   | Protein Target | 1000       | 500       | 500        |
| DrugOOD-EC50-Assay  | Molecules   | Protein Target | 1000       | 500       | 500        |
