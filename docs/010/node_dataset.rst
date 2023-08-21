Datasets for Node-Level Problems
===================================

-----
Cora
-----
`Cora <https://docs.dgl.ai/generated/dgl.data.CoraGraphDataset.html>`_ is a citation network. Each node is a scientific publication,
and its class is the field of the publication. Each node feature is 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. Based on selected six classes (among seven classes) in each dataset, we formulate
three binary classification tasks for Task-IL and three tasks with 2, 4, and 6 classes for Class-IL.

Statistics:

- Nodes: 2,708
- Edges: 10,556
- Number of Node Features: 1,433
- Number of Classes: 7
- Supported Incremental Settings:
   
   + Task-IL with 3 tasks
   + Class-IL with 3 tasks

Citing:

.. code-block::

   @article{sen2008collective,
     title={Collective classification in network data},
     author={Sen, Prithviraj and Namata, Galileo and Bilgic, Mustafa and Getoor, Lise and Galligher, Brian and Eliassi-Rad, Tina},
     journal={AI magazine},
     volume={29},
     number={3},
     pages={93--93},
     year={2008}
   }

-----

----------
Citeseer
----------
`Citeseer <https://docs.dgl.ai/generated/dgl.data.CiteseerGraphDataset.html>`_ is a citation network. Each node is a scientific publication,
and its class is the field of the publication. Each node feature is 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. Based on six classes in each dataset, we formulate
three binary classification tasks for Task-IL and three tasks with 2, 4, and 6 classes for Class-IL.

Statistics:

- Nodes: 3,327
- Edges: 9,104
- Number of Node Features: 3,703
- Number of Classes: 6
- Supported Incremental Settings:
   
   + Task-IL with 3 tasks
   + Class-IL with 3 tasks
   
Citing:

.. code-block::

   @article{sen2008collective,
     title={Collective classification in network data},
     author={Sen, Prithviraj and Namata, Galileo and Bilgic, Mustafa and Getoor, Lise and Galligher, Brian and Eliassi-Rad, Tina},
     journal={AI magazine},
     volume={29},
     number={3},
     pages={93--93},
     year={2008}
   }

-----

----------
CoraFull
----------
`CoraFull <https://docs.dgl.ai/generated/dgl.data.CoraFullDataset.html>`_ is a citation network. Each node is a scientific publication,
and its class is the field of the publication. Each node feature is 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
For CoraFull, we formulate 35 binary classification tasks for Task-IL. 

Statistics:

- Nodes: 19,793
- Edges: 126,842
- Number of Node Features: 8,710
- Number of Classes: 70
- Supported Incremental Settings:
   
   + Task-IL with 35 tasks

Citing:

.. code-block::

   @inproceedings{bojchevski2018deep,
      title={Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking},
      author={Aleksandar Bojchevski and Stephan Günnemann},
      booktitle={ICLR},
      year={2018},
   }

-----

--------------
ogbn-mag
--------------

We extract, from `ogbn-mag <https://ogb.stanford.edu/docs/nodeprop/#ogbn-products>`_ , the citation network between research papers from 2010 to 2019. Each node has 128-dimensional word2vec feature vector. For Task-IL and Class-IL, While the original dataset has 349 node classes indicating fields of studies, we use the 257 classes with 10 or more nodes in validation and test splits. They are divided into 128 groups for Task-IL. Similarly, the number of classes increases by 2 in each task in Class-IL. For Time-IL, we formulate $10$ tasks by constructing tasks with the papers published in the same year. Specifically, the nodes newly revealed in `i`-th task, are the papers published in `2009 + i`. 

Statistics:

- Nodes: 736,389
- Edges: 10,832,542
- Number of Node Features: 128
- Number of Classes: 257 (For Task-IL and Class-IL), 349 (For Time-IL)
- Supported Incremental Settings:
   
   + Task-IL with 128 tasks
   + Class-IL with 128 tasks
   + Time-IL with 10 tasks

Citing:

.. code-block::

   @inproceedings{hu2020open,
     title={Open graph benchmark: datasets for machine learning on graphs},
     author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
     booktitle={NeurIPS},
     year={2020}
   }
   
   @article{wang2020microsoft,
     title={Microsoft academic graph: When experts are not enough},
     author={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
     journal={Quantitative Science Studies},
     volume={1},
     number={1},
     pages={396--413},
     year={2020}
   }

-----

--------------
ogbn-products
--------------
`ogbn-products <https://ogb.stanford.edu/docs/nodeprop/#ogbn-products>`_ is a co-purchase network, where each node
is a product, and its class belongs to 47 categories, which are divided into 9 groups for Class-IL.
The number of classes increase by 5 in each task, and two categories are not used. The node features are extracted from the product descriptions. 

Statistics:

- Nodes: 2,449,029
- Edges: 61,859,140
- Number of Node Features: 100
- Number of Classes: 47
- Supported Incremental Settings:
   
   + Class-IL with 9 tasks

Citing:

.. code-block::

   @inproceedings{hu2020open,
     title={Open graph benchmark: datasets for machine learning on graphs},
     author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
     booktitle={NeurIPS},
     year={2020}
   }
   
   @inproceedings{chiang2019cluster,
     title={Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks},
     author={Chiang, Wei-Lin and Liu, Xuanqing and Si, Si and Li, Yang and Bengio, Samy and Hsieh, Cho-Jui},
     booktitle={KDD},
     year={2019}
   }

-----

---------------
ogbn-proteins
---------------

Nodes in `ogbn-proteins <https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins>`_ are proteins, and edges indicate
meaningful associations between proteins. For each protein, 112 binary classes, which indicate the
presence of 112 functions, are available. Each protein belongs to one among 8 species, which are
used as domains in Domain-IL. Each of the 8 task consists of 112 binary-classification problems.
In our framework, we converted the edge features to the node features by performing mean neighborhood aggregation, as in `the example provided by OGB <https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/proteins>`_.

Statistics:

- Nodes: 132,534
- Edges: 39,561,252
- Number of Node Features: 8
- Number of Classes: 2x112 (112 binary classes)
- Supported Incremental Settings:
   
   + Domain-IL with 8 tasks

.. code-block::

   @inproceedings{hu2020open,
     title={Open graph benchmark: datasets for machine learning on graphs},
     author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
     booktitle={NeurIPS},
     year={2020}
   }
   
   @article{szklarczyk2019string,
     title={STRING v11: protein--protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets},
     author={Szklarczyk, Damian and Gable, Annika L and Lyon, David and Junge, Alexander and Wyder, Stefan and Huerta-Cepas, Jaime and Simonovic, Milan and Doncheva, Nadezhda T and Morris, John H and Bork, Peer and others},
     journal={Nucleic Acids Research},
     volume={47},
     number={D1},
     pages={D607--D613},
     year={2019}
   }

-----


---------------
ogbn-arxiv
---------------
`ogbn-arxiv <https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv>`_ is a citation network, where each node is a
research paper, and its class belongs to 40 subject areas, which are divided into 8 groups for Task-
IL. Similarly, the number of classes increase by 5 in each task in Class-IL. Publication years are
used to form 24 groups for the Time-IL setting.
Specifically, we constructed the first task with the paper published before the year $1998$. For each subsequent `i`-th task, we used the papers published in the year `(1996 + i)`.


Statistics:

- Nodes: 169,343
- Edges: 2,232,486
- Number of Node Features: 128
- Number of Classes: 40
- Supported Incremental Settings:
   
   + Task-IL with 8 tasks
   + Class-IL with 8 tasks
   + Time-IL with 24 tasks

Citing:

.. code-block::

   @inproceedings{hu2020open,
     title={Open graph benchmark: datasets for machine learning on graphs},
     author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
     booktitle={NeurIPS},
     year={2020}
   }
   
   @article{wang2020microsoft,
     title={Microsoft academic graph: When experts are not enough},
     author={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
     journal={Quantitative Science Studies},
     volume={1},
     number={1},
     pages={396--413},
     year={2020}
   }

-----

---------------
twitch
---------------
Nodes in  `twitch <https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv>`_ are users, and edges indicate mutual follower relationship between users. For
each user, there is a label whether the user is joining the affiliate program or not. Each user belongs
to one among 21 broadcasting language groups, which are used as domains in Domain-IL. Using
the binary labels, we formulate 21 binary classification tasks.

Statistics:

- Nodes: 168,114
- Edges: 6,797,557
- Number of Node Features: 4
- Number of Classes: 40
- Supported Incremental Settings:
   
   + Task-IL with 8 tasks
   + Class-IL with 8 tasks
   + Time-IL with 24 tasks

Citing:

.. code-block::

   @misc{rozemberczki2021twitch,
       title = {Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings}, 
       author = {Benedek Rozemberczki and Rik Sarkar},
       year = {2021},
       eprint = {2101.03091},
       archivePrefix = {arXiv},
       primaryClass = {cs.SI}
   }
