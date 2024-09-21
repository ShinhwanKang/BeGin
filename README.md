<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/logo.png" height="200px" width="600px"></p>

[![Latest Release](https://img.shields.io/badge/Latest-v0.4-success)](https://github.com/ShinhwanKang/BeGin/releases)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)


<tr><td colspan="4"> <a href=https://begin.readthedocs.io/>Read the docs</a></td></tr> | <tr><td colspan="4"> <a href=https://arxiv.org/abs/2211.14568>Paper</a></td></tr> | <tr><td colspan="4"> <a href="#Installation">Installation</a></td></tr> | <tr><td colspan="4"> <a href="#Citing-BeGin">Citing Begin</a></td></tr> | <tr><td colspan="4"> <a href=https://docs.google.com/spreadsheets/d/1lS1JHpTAdLTfdMKvhzJ4TFZ6Pap7MYQD__CfNGBdHns/edit?usp=sharing>Detailed Hyperparameter Settings</a></td></tr>


# Overview

**BeGin** is an easy and fool-proof framework for graph continual learning.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/framework.png" width="800px"></p>

Our framework **BeGin** has the following advantages:

- BeGin is easy-to-use. It is easily extended since it is modularized with reusable modules for data processing, algorithm design, training, and evaluation.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/trainer_example.png" width="800px"></p>

- BeGin is fool-proof by completely separating the evaluation module from the learning part, where users implement their own graph CL methods, in order to eliminate potential mistakes in evaluation.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/interaction.png" width="800px"></p>

- BeGin provides 35 benchmark scenarios for graph from 24 real-world datasets, which cover 12 combinations of the incremental settings and the levels of problem. In addition, BeGin provides various basic evaluation metrics for measuring the performances and final evalution metrics designed for continual learning.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/coverage.png" width="600px"></p>

- To the best of our knowledge, we are the first to apply and evaluate parameter-isolation-based methods to graph CL.

## Installation

You can install BeGin with the following command:

```bash
pip install -e .
```

Before running the command, we strongly recommend installing the proper version of `PyTorch`, `DGL`, and `torch-scatter` depending on your CUDA version.

### Dependencies
- `torch>=1.8.1`
- `dgl>=0.6.1`
- `torch-scatter>=2.0.6`
- `torch-sparse>=0.6.9`
- `torch-geometric>=2.0.4`
- `ogb>=1.3.4`
- `dgl-lifesci>=0.3.0`
- `rdkit-pypi>=2022.9.1`

For running some algorithms, you may need the following additional packages:

- `quadprog`
- `cvxpy`
- `qpth`

## Package Usage

The tutorial and documents of BeGin are available at [Here](https://begin.readthedocs.io/).

We also provide some running examples in `examples` directory.

## Citing BeGin

If you use this framework as part of any published research, please consider acknowledging our paper.

```
@article{ko2022begin,
  title={BeGin: Extensive Benchmark Scenarios and An Easy-to-use Framework for Graph Continual Learning},
  author={Ko, Jihoon and Kang, Shinhwan and Shin, Kijung},
  journal={arXiv preprint arXiv:2211.14568},
  year={2022}
}
```
