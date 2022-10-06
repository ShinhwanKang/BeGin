Datasets for Node-Level Problems
===================================

-----
Cora
-----
`Cora <https://docs.dgl.ai/generated/dgl.data.CoraGraphDataset.html>`_ is a citation network. Each node is a scientific publication,
and its class is the field of the publication. Based on selected six classes (among seven classes) in each dataset, we formulate
three binary classification tasks for Task-IL and three tasks with 2, 4, and 6 classes for Class-IL.

Statistics:

- Nodes: 2.708
- Edges: 10.556
- Number of Node Features: 1,433
- Number of Classes: 7
- Supported Incremental Settings:
   
   + Task-IL with 3 tasks
   + Class-IL with 3 tasks

-----

----------
Citeseer
----------
`Citeseer <https://docs.dgl.ai/generated/dgl.data.CiteseerGraphDataset.html>`_ is a citation network. Each node is a scientific publication,
and its class is the field of the publication. Based on six classes in each dataset, we formulate
three binary classification tasks for Task-IL and three tasks with 2, 4, and 6 classes for Class-IL.

Statistics:

- Nodes: 3,327
- Edges: 9,104
- Number of Node Features: 3,703
- Number of Classes: 6
- Supported Incremental Settings:
   
   + Task-IL with 3 tasks
   + Class-IL with 3 tasks
   
-----

----------
CoraFull
----------
`CoraFull <https://docs.dgl.ai/generated/dgl.data.CoraFullDataset.html>`_ is a citation network. Each node is a scientific publication,
and its class is the field of the publication.
For CoraFull, we formulate 35 binary classification tasks for Task-IL. 

Statistics:

- Nodes: 19,793
- Edges: 126,842
- Number of Node Features: 8,710
- Number of Classes: 70
- Supported Incremental Settings:
   
   + Task-IL with 35 tasks
   
-----

--------------
ogbn-products
--------------
`ogbn-products <https://ogb.stanford.edu/docs/nodeprop/#ogbn-products>`_ is a co-purchase network, where each node
is a product, and its class belongs to 47 categories, which are divided into 9 groups for Class-IL.
The number of classes increase by 5 in each task, and two categories are not used.

Statistics:

- Nodes: 2,449,029
- Edges: 61,859,140
- Number of Node Features: 100
- Number of Classes: 47
- Supported Incremental Settings:
   
   + Class-IL with 9 tasks
   
-----

---------------
ogbn-proteins
---------------

Nodes in `ogbn-proteins <https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins>`_ are proteins, and edges indicate
meaningful associations between proteins. For each protein, 112 binary classes, which indicate the
presence of 112 functions, are available. Each protein belongs to one among 8 species, which are
used as domains in Domain-IL. Each of the 8 task consists of 112 binary-classification problems.

Statistics:

- Nodes: 132,534
- Edges: 39,561,252
- Number of Edge Features: 8
- Number of Classes: 2x112 (112 binary classes)
- Supported Incremental Settings:
   
   + Domain-IL with 8 tasks
   
-----


---------------
ogbn-arxiv
---------------
`ogbn-arxiv <https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv>`_ is a citation network, where each node is a
research paper, and its class belongs to 40 subject areas, which are divided into 8 groups for Task-
IL. Similarly, the number of classes increase by 5 in each task in Class-IL. Publication years are
used to form 11 groups for the Time-IL setting.

Statistics:

- Nodes: 169,343
- Edges: 2,232,486
- Number of Node Features: 128
- Number of Classes: 40
- Supported Incremental Settings:
   
   + Task-IL with 8 tasks
   + Class-IL with 8 tasks
   + Time-IL with 11 tasks
