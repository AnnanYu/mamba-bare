import sys
import warnings
import os
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
    HIP_HOME
)



# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "mamba_bare"
FORCE_CXX11_ABI = os.getenv("MAMBA_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_ver = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_ver


def get_hip_version(rocm_dir):

    hipcc_bin = "hipcc" if rocm_dir is None else os.path.join(rocm_dir, "bin", "hipcc")
    try:
        raw_output = subprocess.check_output(
            [hipcc_bin, "--version"], universal_newlines=True
        )
    except Exception as e:
        print(
            f"hip installation not found: {e} ROCM_PATH={os.environ.get('ROCM_PATH')}"
        )
        return None, None

    for line in raw_output.split("\n"):
        if "HIP version" in line:
            rocm_version = parse(line.split()[-1].rstrip('-').replace('-', '+')) # local version is not parsed correctly
            return line, rocm_version

    return None, None


def get_torch_hip_version():

    if torch.version.hip:
        return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))
    else:
        return None


def check_if_hip_home_none(global_option: str) -> None:

    if HIP_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found.  Are you sure your environment has hipcc available?"
    )


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass = {}
ext_modules = []


HIP_BUILD = bool(torch.version.hip)


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

cc_flag = []

if HIP_BUILD:
    check_if_hip_home_none(PACKAGE_NAME)

    rocm_home = os.getenv("ROCM_PATH")
    _, hip_version = get_hip_version(rocm_home)

    if HIP_HOME is not None:
        if hip_version < Version("6.0"):
            raise RuntimeError(
                f"{PACKAGE_NAME} is only supported on ROCm 6.0 and above.  "
                "Note: make sure HIP has a supported version by running hipcc --version."
            )
        if hip_version == Version("6.0"):
            warnings.warn(
                f"{PACKAGE_NAME} requires a patch to be applied when running on ROCm 6.0. "
                "Refer to the README.md for detailed instructions.",
                UserWarning
            )

    cc_flag.append("-DBUILD_PYTHON_PACKAGE")

else:
    check_if_cuda_home_none(PACKAGE_NAME)
    # Check, if CUDA11 is installed for compute capability 8.0

    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )

    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_53,code=sm_53")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_62,code=sm_62")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_72,code=sm_72")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_87,code=sm_87")

    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")


# HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
# torch._C._GLIBCXX_USE_CXX11_ABI
# https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
if FORCE_CXX11_ABI:
    torch._C._GLIBCXX_USE_CXX11_ABI = True

if HIP_BUILD:

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            f"--offload-arch={os.getenv('HIP_ARCHITECTURES', 'native')}",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-fgpu-flush-denormals-to-zero",
        ]
        + cc_flag,
    }
else:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": append_nvcc_threads(
            [
                "-O3",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ptxas-options=-v",
                "-lineinfo",
            ]
            + cc_flag
        ),
    }

ext_modules.append(
    CUDAExtension(
        name="selective_scan_cuda",
        sources=[
            "csrc/selective_scan/selective_scan.cpp",
            "csrc/selective_scan/selective_scan_fwd_fp32.cu",
            "csrc/selective_scan/selective_scan_fwd_fp16.cu",
            "csrc/selective_scan/selective_scan_fwd_bf16.cu",
            "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
            "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
            "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
            "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
            "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
            "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
    )
)



class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        super().run()

setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    packages=find_packages(include=['mamba_core', 'mamba_core.*']),
    author="",
    author_email="",
    description="",

    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    
    python_requires=">=3.8",
    install_requires=[
        "packaging",
        "protobuf<4.24",
        "fsspec==2023.10.0",
        "datasets==2.15.0",
        "aiohttp", # https://github.com/aio-libs/aiohttp/issues/6794
        "dill==0.3.6",
        "multiprocess==0.70.14",
        "huggingface-hub==0.23.4",
        "transformers==4.42.4",
        "einops==0.7.0",
        "ftfy==6.1.3",
        "opt-einsum==3.3.0",
        "pydantic==2.5.3",
        "pydantic-core==2.14.6",
        "pykeops==2.2",
        "python-dotenv==1.0.0",
        "sentencepiece==0.2.0",
        "tokenizers==0.19.1",
        "six==1.16.0",
        "scikit-learn==1.5.1",
        "lm-eval==0.4.1",
        "ninja==1.11.1.1",
        #"flash-attn==2.5.2",
        #"causal-conv1d",
        
        "rich",
        "hydra-core==1.3.2",
        "hydra_colorlog",
        "wandb==0.16.2",
        "lightning-bolts==0.7.0",
        "lightning-utilities==0.10.0",
        "pytorch-lightning==1.8.6",
        "timm"    
    ],
)

