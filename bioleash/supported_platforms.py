import argparse
import os
from enum import Enum

# if TORCH_VERSION envvar is provided, then use this version of torch
# otherwise, fallback to default version
DEFAULT_PYTORCH_VERSION = "2.0"


class SUPPORTED_CUDA_VERSIONS(Enum):
    CUDA_10_2 = "10.2"
    CUDA_11_1 = "11.1"
    CUDA_11_7 = "11.7"


class SUPPORTED_PYTORCH_VERSIONS(Enum):
    TORCH_2_0 = "2.0"
    TORCH_1_13 = "1.13"
    TORCH_1_12 = "1.12"


# a mapping between cuda versions and DSP buildpack builders
CUDA_BUILDPACK_MAPPING = {
    SUPPORTED_CUDA_VERSIONS.CUDA_10_2.value: "ds-rhel7-cuda10-2",
    SUPPORTED_CUDA_VERSIONS.CUDA_11_1.value: "ds-rhel7-cuda11-1",
    SUPPORTED_CUDA_VERSIONS.CUDA_11_7.value: "ds-rhel7-cuda11-7",
}
