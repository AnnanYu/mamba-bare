from setuptools import setup, find_packages



# ensure that torch is installed, and send to torch website if not
try:
    import torch
except ModuleNotFoundError:
    raise ValueError("Please install torch first: https://pytorch.org/get-started/locally/")

_REQUIRED = [
    "packaging",
    "protobuf",
    #"fsspec==2023.10.0",
    #"datasets==2.15.0",
    #"aiohttp", # https://github.com/aio-libs/aiohttp/issues/6794
    #"dill==0.3.6",
    #"multiprocess==0.70.14",
    "huggingface-hub==0.23.4",
    "einops==0.7.0",
    #"ftfy==6.1.3",
    "opt-einsum==3.3.0",
    #"pydantic==2.5.3",
    #"pydantic-core==2.14.6",
    #"pykeops==2.2",
    "python-dotenv==1.0.0",
    "sentencepiece==0.1.99",
    "transformers",
    #"six",
    #"lm-eval",
    "ninja",
    #"flash-attn",
    #"causal-conv1d",
    "rich",
    "hydra-core==1.3.2",
    "hydra_colorlog",
    "wandb",
    #"lightning-bolts",
    #"lightning-utilities",
    "pytorch-lightning",
    "timm"    
]

_OPTIONAL = {
    "dev": [
        "pytest"
    ]
}


setup(
    name='mamba-light',
    version="0.0.1",
    packages=find_packages(include=['src', 'src.*']),
    author="",
    author_email="",
    description="",
    python_requires=">=3.8",
    install_requires=_REQUIRED,
    extras_require=_OPTIONAL,
)
