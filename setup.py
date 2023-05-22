from setuptools import setup, find_packages

core_requires = [
    "torch~=1.11",
    "tensorboardX",
    "torchdiffeq @ git+https://github.com/rtqichen/torchdiffeq",
    "tqdm",
    "pyyaml",
]
exp_requires = [
    "matplotlib",
    "seaborn",
    "torchvision~=0.14",
    "scikit-learn",
    "pygame",
    "pymunk~=5.6.0",
    "POT~=0.9.0"
]

setup(
    name="ncdssm",
    description="Neural Continuous Discrete State Space Models",
    long_description='Neural Continuous Discrete State Space Models presented in the IMCL 2023 paper titled "Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series"',  # noqa
    version="0.0.1",
    install_requires=core_requires,
    extras_require={
        "exp": exp_requires,
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
