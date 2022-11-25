from setuptools import setup, find_packages

setup(
    name='begin',
    version='0.1.0',
    packages=find_packages(include=['begin', 'begin.algorithms.*', 'begin.evaluators.*', 'begin.scenarios.*', 'begin.trainers.*'], exclude=['_build', 'docs', 'dist', 'build', 'begin.egg-info']),
    author='BeGin TEAM',
    install_requires=[
        'dgl>=0.6.1',
        'torch-scatter>=2.0.6',
        'torch-sparse>=0.6.9',
        'torch-geometric>=2.0.4',
        'ogb>=1.3.4',
        'quadprog',
        'cvxpy',
        'qpth',
        'dgllife>=0.2.9',
        'rdkit-pypi'
    ]
)