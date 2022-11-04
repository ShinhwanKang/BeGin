Common framework
======

Our framework ``BeGin`` contains the trainer managing the overall graph continual learning procudure, including preparing the dataloader, training, and validation.
Therefore, users only have to implement novel parts of their methods.

.. Similar to the ``DGLBasicIL``, 
According to the graph problems (e.g., node-, link-, and graph-level), codes of ``trainer`` users extend are different as follows:

.. toctree::
    :maxdepth: 1
    
    node
    link
    graph

Then, the above three codes is inherited a base code as follows:

.. automodule:: code_trainer.common
    :undoc-members:
    :members:
    :private-members: