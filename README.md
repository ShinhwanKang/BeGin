<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/logo.png" height="100px" width="300px"></p>

# Overview

**BeGin** is an easy and fool-proof framework for graph continual learning.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/framework.png" width="800px"></p>

Our framework **BeGin** has the following advantages:

- BeGin is easy-to-use. It is easily extended since it is modularized with reusable modules for data processing, algorithm design, training, and evaluation.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/trainer_example.png" width="800px"></p>

- BeGin is fool-proof by completely separating the evaluation module from the learning part, where users implement their own graph CL methods, in order to eliminate potential mistakes in evaluation.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/interaction.png" width="800px"></p>

- BeGin provides 23 benchmark scenarios for graph from 14 real-world datasets, which cover 12 combinations of the incremental settings and the levels of problem. In addition, BeGin provides various basic evaluation metrics for measuring the performances and final evalution metrics designed for continual learning.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/coverage.png" width="600px"></p>

 

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
- `dgl-lifesci>=0.2.9`
- `rdkit-pypi>=2022.9.1`

For running some algorithms, you may need the following additional packages:

- `quadprog`
- `cvxpy`
- `qpth`

## Package Usage

The tutorial and documents of BeGin are available at https://begin.readthedocs.io/.

We also provide some running examples in `examples` directory.

