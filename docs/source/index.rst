Welcome to BeGin Tutorials and Documentation!
===================================

**BeGin** (**Be**nchmark **G**raph Cont**in**ual Learning) is an *easy* and *fool-proof* framework for graph continual learning.
**BeGin** is *easily extended* since it is *modularized* with reusable module for data processing, algorithm design, validation, and evaluation.

# a Python library for cooks and food lovers
# that creates recipes mixing random ingredients.
# It pulls data from the `Open Food Facts database <https://world.openfoodfacts.org/>`_
# and offers a *simple* and *intuitive* API.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree:: ./000/
   :maxdepth: 1
   :caption: Get Started   

   install
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Datasets
   :hidden:

   010/node_dataset
   010/link_dataset
   010/graph_dataset

.. toctree::
   :maxdepth: 2
   :caption: Dataset Loader
   :hidden:
   :titlesonly:

   020/base
   020/load_node_datset
   020/load_link_datset
   020/load_graph_datset

.. toctree::
   :maxdepth: 2
   :caption: Scenarios
   :hidden:
   :titlesonly:
   :glob:

   030/base
   030/node_cls
   030/link_pred
   030/graph_cls

.. toctree::
   :maxdepth: 2
   :caption: Trainer
   :hidden:
   :titlesonly:
   :glob:

   040/common
   040/node
   040/link
   040/graph

.. toctree::
   :maxdepth: 2
   :caption: Evaluator
   :hidden:
   :titlesonly:
   :glob:

   050/performance
   050/metric



