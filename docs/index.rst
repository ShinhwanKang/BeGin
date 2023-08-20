Welcome to BeGin Tutorials and Documentation!
================================================

**BeGin** is an *easy* and *fool-proof* framework for graph continual learning. First, **BeGin** is *easily extended* since it is modularized with reusable module for data processing, algorithm design, training, validation, and evaluation. Next, **BeGin** is *fool-proof* by completely separating the evaluation module from the learning part, where users implement their own graph CL methods, in order to eliminate potential mistakes in evaluation. In addition, **BeGin** provides 31 benchmark scenarios for graph from 20 real-world datasets, which cover 12 combinations of the incremental settings and the levels of problem. For the details, please refer the contents below.

Contents
------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started   

   000/install
   000/tutorial
   000/load

.. toctree::
   :caption: Datasets

   010/node_dataset
   010/link_dataset
   010/graph_dataset

.. toctree::
   :maxdepth: 1
   :caption: Scenarios

   030/common
   030/node
   030/link
   030/graph
   
.. toctree::
   :maxdepth: 1
   :caption: Trainer

   040/common
   040/node
   040/link
   040/graph

.. toctree::
   :maxdepth: 1
   :caption: Continual Learning Methods

   050/bare
   050/lwf
   050/ewc
   050/mas
   050/gem
   050/twp
   050/ergnn
   050/cgnn
   050/packnet
   050/piggyback
   050/hat
   
.. toctree::
   :maxdepth: 1
   :caption: Evaluator

   060/performance
   060/metric

.. toctree::
   :maxdepth: 1
   :caption: Utils

   070/linear

