from setuptools import setup

REQUIREMENTS = (
    [
        "cloudpickle>=2.2.0",
        "gym==0.21.0",
        "ipdb>=0.13.9",
        "matplotlib>=3.5.3",
        "mujoco-py==2.1.2.14",
        "numpy>=1.23.3",
        "pandas>=1.4.4",
        "pybullet==3.2.5",
        "scipy>=1.9.1",
        "stable-baselines3==1.6.0",
        "termcolor>=2.0.1",
        "torch>=1.12.1",
        "typing_extensions==4.3.0",
        "xmltodict>=0.13.0",
        "pyyaml>=6.0",
        "tensorboard>=2.10.1",
        "gurobipy==10.0.0",
    ]
)

setup(
    install_requires=REQUIREMENTS
)
