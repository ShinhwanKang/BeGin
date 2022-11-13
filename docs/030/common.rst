Common framework
====================

Our framework ``BeGin`` provides a sceanrio loader responsible for communicating with user code (i.e., the training part) to perform a benchmark under a sesired incremental setting.

According to the graph problems, it will be different that users need. 
Therefore, we provide 4 scenario loaders and a base framework for further implementation. 

We provides the implemented scenario loaders as follows:

- `Node-level problems <./node.html>`_

- `Link-level problems <./link.html>`_

- `Graph-level problems <./graph.html>`_

The base framework is as follows:

.. autoclass:: begin.scenarios.common.BaseScenarioLoader
    :members:
    :private-members: