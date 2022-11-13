Common framework
====================

Our framework ``BeGin`` contains the trainer managing the overall graph continual learning procudure, including preparing the dataloader, training, and validation.
Therefore, users only have to implement novel parts of their methods.

According to the graph problems (e.g., node-, link-, and graph-level), codes of ``trainer`` users extend are different as follows:


- `Node-level problems <./node.html>`_

- `Link-level problems <./link.html>`_

- `Graph-level problems <./graph.html>`_

.. .. toctree::
..     :maxdepth: 1
    
..     node
..     link
..     graph

Then, the above three codes is inherited a base code as follows:

.. note::

   In the framework, all task-specific trainers assumes that `AdaptiveLinear` is used as the `classifier` of the model.
   See `here <../070/linear.html>`_ for the details of the module.

.. automodule:: begin.trainers.common
    :members:
    :private-members: