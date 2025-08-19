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

> [Unifying Unsupervised Graph-Level Out-of-Distribution Detection and Anomaly Detection: A Benchmark](https://arxiv.org/abs/2406.15523)
> 

UB-GOLD provides a fair and comprehensive platform to evaluate 18 existing unsupervised GLAD and GLOD works on 4 types of datasets and facilitates future GLAD/GLOD research.

------
<div align="center">
  <img src="https://raw.githubusercontent.com/UB-GOLD/UB-GOLD/main/Image/intro.jpg" width="900px"/>
</div>

------

<div align="center">
  <img src="https://raw.githubusercontent.com/UB-GOLD/UB-GOLD/main/Image/UBGOLD.jpg" width="900px"/>
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

You can easily create a virtual environment with all the required libraries for this project by running:  
`conda env create -n UBGOLD -f environment.yml`  

If errors occur during the above step, you may also install the dependencies using pip. Before using pip, it is strongly recommended to manually select the appropriate PyTorch version (compatible with your Python and CUDA versions). In this example, the Python version is 3.8.10 and the CUDA version is 11.8. Thus, the manually downloaded PyTorch and its extension packages are as follows:  
- `torch-2.1.0+cu118-cp38-cp38-linux_x86_64.whl`  
- `torch_scatter-2.1.2+pt21cu118-cp38-cp38-linux_x86_64.whl`  
- `torch_sparse-0.6.18+pt21cu118-cp38-cp38-linux_x86_64.whl`  
- `torchvision-0.15.1+cu118-cp38-cp38-linux_x86_64.whl`  

If you use a different CUDA version, please refer to the PyTorch and Pyg websites for the appropriate versions.

For additional `torch_geometric` extensions, you can download them from [this link](https://data.pyg.org/whl/).  
For PyTorch packages, users in China can download them from the [Nanjing University mirror site](https://mirrors.nju.edu.cn/pytorch/whl/).  

After manually configuring the PyTorch-related packages, run the following command to install the remaining Python libraries:  
`pip install -r requirements.txt`  

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
1. Intrinsic Anomaly: [TOX21](https://tripod.nih.gov/tox21/challenge/data.jsp#)
   - Tox21\_p53, Tox21\_HSE, Tox21\_MMP,Tox21\_PPAR-gamma
2. Inter-Dataset Shift & Class-based Anomaly: [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/) 
    - COLLAB, IMDB-BINARY, REDDIT-BINARY, ENZYMES, PROTEINS
    - DD, BZR, AIDS, COX2, NCI1, DHFR
3. Inter-Dataset Shift: [OGB](https://github.com/snap-stanford/ogb)
   - BBBP, BACE, CLINTOX, LIPO, FREESOLV
   - TOXCAST, SOL, MUV, TOX21, SIDER
4. Intra-Dataset Shift: [DrugOOD](https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC) 
   - IC50 (SIZE, SCAFFOLD, ASSAY)
   - EC50 (SIZE, SCAFFOLD, ASSAY)
5. Intra-Dataset Shift: GOOD 
   - [GOODHIV](https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view)
   - [GOODZINC](https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view)
  
<div align="left">
  <img src="https://github.com/UB-GOLD/UB-GOLD/blob/main/Image/Dataset.jpg" width="600px"/>
</div>
     
## Quick Start

```
git clone https://github.com/UB-GOLD/UB-GOLD.git
cd ./UBGOLD
```
**With Default Hyperparameters**

The main experiments:
```
python benchmark/mymain.py 

OR

cp ./benchmark/Source/GOOD-D.sh .

bash GOOD-D.sh
```
The near-far OOD detection experiments:

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
If you find our repository useful for your research, please consider citing our paper:
```
@article{wang2024unifying,
  title={Unifying Unsupervised Graph-Level Anomaly Detection and Out-of-Distribution Detection: A Benchmark},
  author={Wang, Yili and Liu, Yixin and Shen, Xu and Li, Chenyu and Ding, Kaize and Miao, Rui and Wang, Ying and Pan, Shirui and Wang, Xin},
  journal={arXiv preprint arXiv:2406.15523},
  year={2024}
}
```
## Reference
UB-GOLD has completed the approaches related to unsupervised Graph detection for Tables 1 and 2. As for some special scenarios in Tables 3-5, we will gradually adapt them to this benchmark.

### Table 1: Unsupervised GLAD 

| **ID** | **Paper** | **Category** | **Conference** |
|--------|---------|:----------:|:--------------:|
| 1      | [On using classification datasets to evaluate graph outlier detection: Peculiar observations and new insights](https://arxiv.org/pdf/2012.12931.pdf)    |   GLAD   |  Big Data 2023   |
| 2      | [Deep graph-level anomaly detection by glocal knowledge distillation ](https://arxiv.org/pdf/2112.10063.pdf)|    GLAD    |   WSDM 2022    |
| 3      |[Raising the bar in graph-level anomaly detection](https://arxiv.org/pdf/2205.13845.pdf) |  GLAD  |    IJCAI 2022    |
| 4      | [Towards self-interpretable graph-level anomaly detection](https://proceedings.neurips.cc/paper_files/paper/2023/file/1c6f06863df46de009a7a41b41c95cad-Paper-Conference.pdf)  |   GLAD   |  NeurIPS 2023  |
| 5      | [Deep graph level anomaly detection with contrastive learning](https://www.nature.com/articles/s41598-022-22086-3) |   GLAD   | Nature scientific reports 2022 |
| 6      | [CVTGAD: Simplified Transformer with Cross-View Attention for Unsupervised Graph-Level Anomaly Detection](https://link.springer.com/chapter/10.1007/978-3-031-43412-9_11) |  GLAD  |  ECML-PKDD 2023    |

### Table 2:  Unsupervised GLOD 
| **ID** | **Paper** | **Category** | **Conference** |
|--------|---------|:----------:|:--------------:|
| 7      | [Good-d: On unsupervised graph out-of-distribution detection ](https://arxiv.org/pdf/2211.04208.pdf)|  GLOD   |   WSDM 2023    |
| 8     |[ Graphde: A generative framework for debiased learning and out-of-distribution detection on graphs](https://proceedings.neurips.cc/paper_files/paper/2022/file/c34262c35aa5f8c1a091822cbb2020c2-Paper-Conference.pdf) |  GLOD    |    NeurIPS 2022    |
| 9   | [A Data-centric Framework to Endow Graph Neural Networks with Out-Of-Distribution Detection Ability](http://shichuan.org/doc/150.pdf) |  GLOD  |    KDD 2023   |
| 10    | [GOODAT: Towards Test-time Graph Out-of-Distribution Detection](https://arxiv.org/pdf/2401.06176v1.pdf) |  GLOD  |    AAAI 2024   |

------

### Table 3: Other Approaches (Supervised Setting)

| **ID** | **Paper** | **Category** | **Conference** |
|--------|---------|:----------:|:--------------:|
| 1      | [Dual-discriminative graph neural network for imbalanced graph-level anomaly detection](https://proceedings.neurips.cc/paper_files/paper/2022/file/98a625423070cfc6ae3d82d4b59408a0-Paper-Conference.pdf)|  GLAD   |    NeurIPS 2022    |
| 2      | [Towards graph-level anomaly detection via deep evolutionary mapping](https://dl.acm.org/doi/pdf/10.1145/3580305.3599524) |  GLAD   |    KDD 2023    |
| 3      |[Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection](https://arxiv.org/pdf/2310.02861.pdf)| GLAD   |   ICLR 2024    |
| 4      | [SGOOD: Substructure-enhanced Graph-Level Out-of-Distribution Detection ](https://arxiv.org/pdf/2310.10237.pdf)|  GLOD  |    Arxiv 2023    |

### Table 4: Other Approaches (Scenario Limitations (limited to molecular graphs))
| **ID** | **Paper** | **Category** | **Conference** |
|--------|---------|:----------:|:--------------:|
| 5   |[Optimizing OOD Detection in Molecular Graphs: A Novel Approach with Diffusion Models](https://arxiv.org/pdf/2404.15625) |  GLOD   |    KDD 2024    |



