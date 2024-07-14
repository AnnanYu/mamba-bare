
** Same base code to train MAMBA on NLP tasks **

This repository is a naked version of the [based](https://github.com/HazyResearch/based/tree/main) repository for training mamba. 

We created this repository because we had some issues with getting [based](https://github.com/HazyResearch/based/tree/main) to run with CUDA 12.4.


## Installation

Setup conda enviroment and install torch. We test this repository using `python=3.12.4` and the Pytorch Preview (Nightly) build with CUDA 12.4.


```bash

# create fresh conda enviroment
conda create -n mamba python=3.12.4

# activate mamba enviroment
conda activate mamba

# install latest torch build with cuda 12.4
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# 


```


