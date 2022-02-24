# (MMFDAs) Multimodal Fusion Dense Autoencoders for Identifying Disease Subtypes

### Introduction

This repository is for our submitted paper for MDPI-Biology '[Integration of Multimodal Data from Disparate Sources for Identifying Disease Subtypes]'. 

### Installation
This repository is based on PyTorch 1.4 and CUDA 10.0. 

For installing PyTorch combining with the CUDA toolkit, please follow the official instructions in [here](https://https://pytorch.org/). The code is tested under PyTorch 1.4 and Python 3.6 on Ubuntu 18.04.

### Usage

1. Download this repository. And go to folder:
```shell
   cd code
   ```
2. Download the data [here](https://drive.google.com/file/d/1kS8I1WfOtr3ilRYpewg76p4LyyOP2vRa/view?usp=sharing) and put in the data folder. An illustration of the folder organization is:
   ```shell
   -root
   --data
   ---gbm
   ---laml
   ---paad
   ``` 

4. Train and test the model.\
   Run complete fusion model (CFA in the paper):
   ```shell
   python main_aec.py
   ```
   Run incomplete fusion model (IFA in the paper):
   ```shell
   python main_aec.py --pred_missing
   ```
   Run complete fusion model with 2 existing modalities (CFA-2M in the paper):
   ```shell
   python main_aec_2m.py
   ```
   Run single modality model (SMA in the paper), change which modality to use `m_use = m*` in `main_aec_single.py`:
   ```shell
   python main_aec_single.py
   ```




