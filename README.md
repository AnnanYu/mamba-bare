
# **A bare repository to train MAMBA on NLP tasks**

This repository combines scripts from the [based](https://github.com/HazyResearch/based/tree/main) and [Mamba](https://github.com/state-spaces/mamba/tree/main) repositories to train MAMBA models on NLP tasks.

## Installation

Setup conda enviroment and install torch. We test this repository using `python=3.12.4` and the Pytorch Preview (Nightly) `python=2.5.0.dev20240714+cu124` build with CUDA 12.4.


```bash

# create fresh conda enviroment
conda create -n mamba python=3.12.4

# activate mamba enviroment
conda activate mamba

# install latest torch build with cuda 12.4
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# install packaging
pip3 install packaging

# clone the repository
git clone https://github.com/erichson/mamba-bare

# install based package
cd mamba-bare
pip3 install -e .


```


To train a new model with CUDA 12.4, we need to install a few additional packages from scratch:

```python

# install apex
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cuda_ext --cpp_ext # modify the setup.py and comment out lines 39-47 to avoid CUDA driver checks.
cd ..


# install falsh-attention (we need to compile from source to make it work with CUDA 12.4 --- this will take a while)
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install # this takes a while ... have a coffee :)
cd ..

# install causal conv1d (we need to compile from source to make it work with CUDA 12.4  --- this will take a while)
git clone https://github.com/Dao-AILab/causal-conv1d
cd causal-conv1d
python setup.py install ## this takes a while ...
cd ..

```



Now, you can (hopefully) train a mamba model on the WikiText103 language modeling data using the following script:
```
cd train/
python run.py experiment=example/mamba-360m trainer.devices=8
```





This project borrows code from a several open source projects:
- [FlashAttention](https://github.com/Dao-AILab/flash-attention). 
- [Mamba](https://github.com/state-spaces/mamba/tree/main).
- [based](https://github.com/HazyResearch/based/tree/main).

