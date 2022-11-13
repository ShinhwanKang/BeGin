Datasets for Graph-Level Problems
===================================

----------
MNIST
----------
Images in `MNIST <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.GNNBenchmarkDataset>`_ are converted to
graphs of super-pixels. There are 10 classes of graphs, and they are partitioned into 5 groups,
which are used separately for Task-IL and accumulated for Class-IL.

Statistics:

- Number of Graphs: 55,000
- Average Number of Nodes: 70.6
- Average Number of Edges: 564.5
- Number of Node Features: 3
- Number of Classes: 10
- Supported Incremental Settings:
   
   + Task-IL with 5 tasks
   + Class-IL with 5 tasks


-----

----------
CIFAR10
----------
Images in `CIFAR10 <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.GNNBenchmarkDataset>`_ are converted to
graphs of super-pixels. There are 10 classes of graphs, and they are partitioned into 5 groups,
which are used separately for Task-IL and accumulated for Class-IL.

Statistics:

- Number of Graphs: 45,000
- Average Number of Nodes: 117.6
- Average Number of Edges: 941.2
- Number of Node Features: 5
- Number of Classes: 10
- Supported Incremental Settings:
   
   + Task-IL with 5 tasks
   + Class-IL with 5 tasks

-----

------------
ogbg-molhiv
------------

Graphs in  `ogbg-molhiv <https://ogb.stanford.edu/docs/graphprop/#ogbg-mol>`_ are molecules
consisting of atoms and their chemical bonds. The binary class of each graph indicates whether
the molecule inhibits HIV virus replication or not. For Domain-IL, we divide molecules into 10
groups based on structural similarity by the scaffold splitting procedure.

Statistics:

- Number of Graphs: 41,127
- Average Number of Nodes: 25.5
- Average Number of Edges: 27.5
- Number of Node Features: 9
- Number of Edge Features: 3
- Number of Classes: 2
- Supported Incremental Settings:
   
   + Domain-IL with 10 tasks
   
   
-----

----------
NYC-Taxi
----------

Each graph in `NYC-Taxi <https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page>`_ shows the amount of taxi traffic between locations in New York City
during an hour in 2021. Specifically, nodes are locations, and there exist a directed edge between
two nodes if there existed a taxi customer between them during an hour. The number of such
customers is used as the edge weight. The date and time of the corresponding taxi traffic are
used to partition the graphs into 12 groups for Time-IL. The binary class of each graph indicates
whether it indicates taxi traffic on weekdays (Mon.-Fri.) or weekends (Sat.-Sun.).

Statistics:

- Number of Graphs: 8,760
- Average Number of Nodes: 265.0
- Average Number of Edges: 1597.8 
- Number of Node Features: 7
- Number of Edge Features: 1
- Number of Classes: 2
- Supported Incremental Settings:
   
   + Time-IL with 12 tasks
