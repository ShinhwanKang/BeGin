<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/logo.png" height="100px" width="300px"></p>

# Overview

**BeGin** is an easy and fool-proof framework for graph continual learning.

Our framework **BeGin** has the following advantages:

- BeGin is easy-to-use. It is easily extended since it is modularized with reusable modules for data processing, algorithm design, training, and evaluation.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/framework.png" width="800px"></p>

- BeGin is fool-proof by completely separating the evaluation module from the learning part, where users implement their own graph CL methods.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/interaction.png" width="800px"></p>

- BeGin provides 23 benchmark scenarios for graph from 14 real-world datasets, which cover 12 combinations of the incremental settings and the levels of problem. In addition, BeGin provides various basic evaluation metrics for measuring the performances and final evalution metrics designed for continual learning.

<p align="center"><img src="https://github.com/ShinhwanKang/BeGin/raw/main/static/coverage.png" width="600px"></p>

 

## Dependencies
- `PyTorch>=1.8.1`
- `DGL>=0.6.1`
- `torch-scatter>=2.0.6`
- `torch-sparse>=0.6.9`
- `torch-geometric>=2.0.4`
- `ogb>=1.3.4`
- `quadprog>=0.1.11`
- `cvxpy>=1.2`
- `qpth>=0.0.15`
- `dgl-lifesci>=0.2.9`
- `rdkit-pypi>=2022.9.1`

## Running Benchmarks

We provide some running examples in `examples` directory.

See `begin` directory to see the implementations, and visit https://begin.readthedocs.io/ for the documents.

