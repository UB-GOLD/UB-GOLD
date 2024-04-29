# (GOD2Bench) Unifying Graph-Level Anomaly and OOD Detection: A Benchmark


This is the official implementation of the following paper:

> [ Unifying Graph-Level Anomaly and OOD Detection: A Benchmark](https://arxiv.org/abs/2306.12251)
> 
> Submited to NeurIPS 2024 Datasets and Benchmarks Track

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

```shell
# Create and activate a new Conda environment named 'GOD2Bench'
conda create -n GOD2Bench
conda activate GOD2Bench

# Install Pytorch and Pyg with CUDA 11.8 support
# If your use a different CUDA version, please refer to the PyTorch and Pyg websites for the appropriate versions.

pip install -r requirements.txt
```
