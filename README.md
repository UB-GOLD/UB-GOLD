<div align="center">
  <img src="https://github.com/UB-GOLD/UB-GOLD/blob/main/Image/Show.jpg" alt="UB-GOLD"  width="500px"/>
</div>

------

<p align="center">
  <a href="#UB-GOLD">Overview</a> •
  <a href="#Environment-setup">Environment</a> •
  <a href="#dataset-preparation">Dataset</a> •
  <a href="#Quick-start">Start</a> •
  <a href="https://github.com/UB-GOLD/UB-GOLD?tab=MIT-1-ov-file#readme">License</a> •
  <a href="#citation">Citation</a> 
</p>

## UB-GOLD

This is the official implementation of the following paper:

> [Unifying Unsupervised Graph-Level Out-of-Distribution Detection and Anomaly Detection: A Benchmark](https://arxiv.org/abs/2306.122)
> 
> Submitted to NeurIPS 2024 Datasets and Benchmarks Track

UB-GOLD provides a fair and comprehensive platform to evaluate existing unsupervised GLAD and GLOD works on 4 types of datasets and facilitates future GLAD/GLOD research.
<div align="center">
  <img src="https://github.com/UB-GOLD/UB-GOLD/blob/main/Image/intro.jpg" width="900px"/>
</div>

## Environment Setup


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

pip install *

# Install Pytorch and Pyg with CUDA 11.8 support, For the versions of other dependencies,
# see requirements.txt.
# If you use a different CUDA version, please refer to the PyTorch and Pyg websites
# for the appropriate versions.
```

## Codebase Folder

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

## Dataset Preparation

The data used in UB-GOLD should be downloaded in './data', and they can be downloaded from the following sources:
1. Intrinsic Anomaly: [TOX21](https://tripod.nih.gov/tox21/challenge/data.jsp#) (Intrinsic Anomaly)
   - Tox21\_p53, Tox21\_HSE, Tox21\_MMP,Tox21\_PPAR-gamma
2. Inter-Dataset Shift & Class-based Anomaly: [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/) 
    - COLLAB, IMDB-BINARY, REDDIT-BINARY, ENZYMES, PROTEINS
    - DD, BZR, AIDS, COX2, NCI1, DHFR
3. Inter-Dataset Shift: [OGB](https://github.com/snap-stanford/ogb) (It can be downloaded automatically) 
   - BBBP, BACE, CLINTOX, LIPO, FREESOLV
   - TOXCAST, SOL, MUV, TOX21,SIDER
4. Intra-Dataset Shift: [DrugOOD](https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC) 
   - IC50 (SIZE,SCAFFOLD,ASSAY)
   - EC50 (SIZE,SCAFFOLD,ASSAY)
5. Intra-Dataset Shift: GOOD 
   - [GOODHIV](https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view)
   - [GOODZINC](https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view)
  
<div align="left">
  <img src="https://github.com/UB-GOLD/UB-GOLD/blob/main/Image/Dataset.jpg" width="600px"/>
</div>
     
## Quick Start

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



## Supported Methods  (16 Methods)

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


## Citation
