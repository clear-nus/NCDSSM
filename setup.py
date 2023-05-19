from setuptools import setup, find_packages

core_requires = [
    "torch~=1.11",
    "tensorboardX",
    "torchdiffeq @ git+https://github.com/rtqichen/torchdiffeq",
    "tqdm",
    "pyyaml",
]
plot_requires = ["matplotlib", "seaborn", "torchvision~=0.14"]
pymunk_requires = ["pygame", "pymunk~=5.6.0", "POT~=0.9.0"]
all_requires = plot_requires + pymunk_requires

setup(
    name="ncdssm",
    description="Neural Continuous Discrete State Space Models",
    long_description='Neural Continuous Discrete State Space Models presented in the IMCL 2023 paper titled "Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series"',  # noqa
    version="0.0.1",
    install_requires=core_requires,
    extras_require={
        "plot": plot_requires,
        "pymunk": pymunk_requires,
        "all": all_requires,
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
