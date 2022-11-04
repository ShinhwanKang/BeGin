Common framework
======

Our framework ``BeGin`` provides a sceanrio loader responsible for communicating with user code (i.e., the training part) to perform a benchmark under a sesired incremental setting.

According to the graph problems, it will be different that users need. 
Therefore, we provide 4 scenario loaders and a base framework for further implementation. 

We provides the implemented scenario loaders as follows:

.. toctree::
    :maxdepth: 1
    
    node
    link
    graph



.. Our benchmark scenarios are based on various node-, link-, graph-level problems.
.. The problems are defined `Here <https://www.naver.com/>`_.



.. We prov...

.. The `DGLBasicIL` ...

The base framework is as follows:

.. autoclass:: code_scenario.common.DGLBasicIL
    :members:
    :undoc-members:
    :private-members: