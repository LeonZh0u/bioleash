import os

from setuptools import find_namespace_packages, setup

from supported_platforms import DEFAULT_PYTORCH_VERSION

pytorch_version = os.getenv("PYTORCH_VERSION", DEFAULT_PYTORCH_VERSION)

setup(
    name="bloomberg.lzhou286.bioleash",
    version="0.0.1",
    url="https://bbgithub.dev.bloomberg.com/lzhou286/bioleash.git",
    description="Bio Leash",
    packages=find_namespace_packages(
        include=[
            "bloomberg.lzhou286.bioleash",
            "bloomberg.lzhou286.bioleash.*",
        ]
    ),
    # torch2.0.1 / torch2.1.0
    install_requires=[
        "urllib3==1.26.16",
        "datasets",
        "numpy==1.23.5",
        # "torch @ https://dep-server.buildpacks.dev.bloomberg.com/resources/wheels/torch/torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl",
        "pytorch_triton @ https://dep-server.buildpacks.dev.bloomberg.com/resources/wheels/torch/pytorch_triton-2.1.0+3c400e7818-cp310-cp310-linux_x86_64.whl",
        "torch @ https://artprod.dev.bloomberg.com/artifactory/api/pypi/python-hpc-wheels/torch/2.1.2+cu121/torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl",
        # "triton @ https://dep-server.buildpacks.dev.bloomberg.com/resources/wheels/torch/triton-2.0.0-1-cp38-cp38-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "transformers==4.31.0",
        "polars",
        "scikit-learn",
        "torch",
        "torchmetrics",
        "lightning",
        "pyarrow",
        "protobuf==3.20.1",
        "loguru",
        "ninja",
        "py3nvml",
        "optimum",
        "bloomberg.ai.remoteio[s3]",
    ],
    # torch1.9
    # install_requires=[
    #     "datasets",
    #     "numpy",
    #     "torch==1.9.1",
    #     "transformers==4.28.1",
    #     "protobuf==3.20.1",
    #     "loguru",
    #     "ninja",
    # ],
    entry_points={
        "console_scripts": [],
    },
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)
