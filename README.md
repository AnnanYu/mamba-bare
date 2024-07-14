
** Same base code to train MAMBA on NLP tasks **

This repository is a naked version of the [based](https://github.com/HazyResearch/based/tree/main) repository for training mamba. 

We created this repository because we had some issues with getting [based](https://github.com/HazyResearch/based/tree/main) to run with CUDA 12.4.


## Installation

Setup conda enviroment and install torch. We test this repository using `python=3.12.4` and the Pytorch Preview (Nightly) `python=2.5.0.dev20240714+cu124` build with CUDA 12.4.


```bash

# create fresh conda enviroment
conda create -n mamba python=3.12.4

# activate mamba enviroment
conda activate mamba

# install latest torch build with cuda 12.4
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# clone the repository
git clone https://github.com/erichson/mamba-light.git

# install based package
cd mamba-light
pip install -e .


```


To train a new model with CUDA 12.4, we need to install a few additional packages from scratch:

```python

# install apex
git clone https://github.com/NVIDIA/apex
cd apex

# modify the setup.py and comment out the part that checks the CUDA driver.

pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
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


# install mamba package
#git clone https://github.com/state-spaces/mamba
#cd mamba
#pip install -e .
#cd ..

```



Now, you can (hopefully) train a mamba model on the WikiText103 language modeling data using the following script:
```
cd train/
python run.py experiment=example/mamba-360m trainer.devices=8
```





This project was made possible by a number of other open source projects; please cite if you use their work! Notably:
- Our training code and sliding window implementation are based on Tri Dao's [FlashAttention](https://github.com/Dao-AILab/flash-attention). 
- We use the conv1d kernel from [Mamba](https://github.com/state-spaces/mamba/tree/main).
- We integrated the causal dot product kernel from [Fast Transformers](https://github.com/idiap/fast-transformers).
- We integrated the based kernels from [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention).


