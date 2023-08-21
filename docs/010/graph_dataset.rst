Datasets for Graph-Level Problems
===================================

----------
MNIST
----------
Images in `MNIST <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.GNNBenchmarkDataset>`_ are converted to
graphs of super-pixels. There are 10 classes of graphs, and they are partitioned into 5 groups,
which are used separately for Task-IL and accumulated for Class-IL. According to the original paper, they used SLIC super-pixels as nodes and build a k-nearest neighbor adjacency matrix to generate edges.

Statistics:

- Number of Graphs: 55,000
- Average Number of Nodes: 70.6
- Average Number of Edges: 564.5
- Number of Node Features: 3
- Number of Classes: 10
- Supported Incremental Settings:
   
   + Task-IL with 5 tasks
   + Class-IL with 5 tasks

Citing:

.. code-block::

   @article{dwivedi2020benchmarking,
     title={Benchmarking graph neural networks},
     author={Dwivedi, Vijay Prakash and Joshi, Chaitanya K and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
     journal={arXiv preprint arXiv:2003.00982},
     year={2020}
   }
   
   @article{achanta2012slic,
     title={SLIC superpixels compared to state-of-the-art superpixel methods},
     author={Achanta, Radhakrishna and Shaji, Appu and Smith, Kevin and Lucchi, Aurelien and Fua, Pascal and S{\"u}sstrunk, Sabine},
     journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
     volume={34},
     number={11},
     pages={2274--2282},
     year={2012},
     publisher={IEEE}
   }

-----

----------
CIFAR10
----------
Images in `CIFAR10 <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.GNNBenchmarkDataset>`_ are converted to
graphs of super-pixels. There are 10 classes of graphs, and they are partitioned into 5 groups,
which are used separately for Task-IL and accumulated for Class-IL. According to the original paper, they used SLIC super-pixels as nodes and build a k-nearest neighbor adjacency matrix to generate edges.

Statistics:

- Number of Graphs: 45,000
- Average Number of Nodes: 117.6
- Average Number of Edges: 941.2
- Number of Node Features: 5
- Number of Classes: 10
- Supported Incremental Settings:
   
   + Task-IL with 5 tasks
   + Class-IL with 5 tasks

Citing:

.. code-block::

   @article{dwivedi2020benchmarking,
     title={Benchmarking graph neural networks},
     author={Dwivedi, Vijay Prakash and Joshi, Chaitanya K and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
     journal={arXiv preprint arXiv:2003.00982},
     year={2020}
   }
   
   @article{achanta2012slic,
     title={SLIC superpixels compared to state-of-the-art superpixel methods},
     author={Achanta, Radhakrishna and Shaji, Appu and Smith, Kevin and Lucchi, Aurelien and Fua, Pascal and S{\"u}sstrunk, Sabine},
     journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
     volume={34},
     number={11},
     pages={2274--2282},
     year={2012},
     publisher={IEEE}
   }

-----

--------------
Aromaticity
--------------

Graphs in `Aromaticity <https://lifesci.dgl.ai/api/data.html#pubmed-aromaticity>`_ are molecules consisting of atoms and their chemical bonds.
The original dataset contains labels representing the number of aromatic atoms in each molecule.
We divide molecules into 30 groups based on the labels and formulate Task-IL and Class-IL settings with 10 tasks. Since there is no external node feature, we use in-degrees and out-degrees as node features.

Statistics:

- Number of Graphs: 3,868
- Average Number of Nodes: 29.7
- Average Number of Edges: 65.4
- Number of Node Features: 0
- Number of Classes: 30
- Supported Incremental Settings:
   
   + Task-IL with 10 tasks
   + Class-IL with 10 tasks

Citing:

.. code-block::

   @article{wu2018moleculenet,
     title={MoleculeNet: a benchmark for molecular machine learning},
     author={Wu, Zhenqin and Ramsundar, Bharath and Feinberg, Evan N and Gomes, Joseph and Geniesse, Caleb and Pappu, Aneesh S and Leswing, Karl and Pande, Vijay},
     journal={Chemical science},
     volume={9},
     number={2},
     pages={513--530},
     year={2018},
     publisher={Royal Society of Chemistry}
   }
   
   @article{xiong2019pushing,
     title={Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism},
     author={Xiong, Zhaoping and Wang, Dingyan and Liu, Xiaohong and Zhong, Feisheng and Wan, Xiaozhe and Li, Xutong and Li, Zhaojun and Luo, Xiaomin and Chen, Kaixian and Jiang, Hualiang and others},
     journal={Journal of Medicinal Chemistry},
     volume={63},
     number={16},
     pages={8749--8760},
     year={2019}
   }


-----

------------
ogbg-molhiv
------------

Graphs in  `ogbg-molhiv <https://ogb.stanford.edu/docs/graphprop/#ogbg-mol>`_ are molecules
consisting of atoms and their chemical bonds. The binary class of each graph indicates whether
the molecule inhibits HIV virus replication or not. For Domain-IL, we divide molecules into 20
groups based on structural similarity by the scaffold splitting procedure.
Input node features are 9-dimensional, containing atomic number and chirality, and edge features are 3-dimensional indicating the types of bonds.

Statistics:

- Number of Graphs: 41,127
- Average Number of Nodes: 25.5
- Average Number of Edges: 27.5
- Number of Node Features: 9
- Number of Edge Features: 3
- Number of Classes: 2
- Supported Incremental Settings:
   
   + Domain-IL with 20 tasks

Citing:

.. code-block::

   @article{wu2018moleculenet,
     title={MoleculeNet: a benchmark for molecular machine learning},
     author={Wu, Zhenqin and Ramsundar, Bharath and Feinberg, Evan N and Gomes, Joseph and Geniesse, Caleb and Pappu, Aneesh S and Leswing, Karl and Pande, Vijay},
     journal={Chemical science},
     volume={9},
     number={2},
     pages={513--530},
     year={2018},
     publisher={Royal Society of Chemistry}
   }
   
   @misc{landrum2006rdkit,
     title={RDKit: Open-source cheminformatics},
     author={Landrum, Greg and others},
     year={2006}
   }
   
   @inproceedings{hu2020open,
     title={Open graph benchmark: datasets for machine learning on graphs},
     author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
     booktitle={NeurIPS},
     year={2020}
   }

   
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
The node features indicate the position of the locations, among "Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island", and "Unknown".

Statistics:

- Number of Graphs: 8,760
- Average Number of Nodes: 265.0
- Average Number of Edges: 1597.8 
- Number of Node Features: 7
- Number of Edge Features: 1
- Number of Classes: 2
- Supported Incremental Settings:
   
   + Time-IL with 12 tasks

.. code-block::

   @misc{nyctaxi,
     title={TLC Trip Record Data},
     author={{NYC Taxi \& Limousine Commission}},
     howpublished = {https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page},
   }


-----

----------
ogbg-ppa
----------

Graphs in `ogbg-ppa <https://ogb.stanford.edu/docs/graphprop/#ogbg-ppa>`_ are protein-protein interactions. For Domain-IL, we formulate multi-class classification problem with $37$ classes to predict what taxonomic groups of species the graph comes from. The dataset is sampled so that there are $11$ species for each taxonomic group and $100$ graphs for each species. We formulate $11$ tasks, and each task was formulated to contain graphs of exactly one species per group so that there is no duplicated graph among the tasks. Since there is no external node feature, we use in-degrees
and out-degrees as node features. According to OGB, the edges are associated with 7-dimensional features, where each element takes a value between 0 and 1 and represents the approximate confidence of a particular type of protein protein association such as gene co-occurrence, gene fusion events, and co-expression.

Statistics:

- Number of Graphs: 40,700
- Average Number of Nodes: 243.4
- Average Number of Edges: 2266.1 
- Number of Node Features: 2
- Number of Edge Features: 7
- Number of Classes: 37
- Supported Incremental Settings:
   
   + Domain-IL with 11 tasks

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
   
   @techreport{hug2016new,
     title={A new view of the tree of life. Nature Microbiology, 1 (5), 16048},
     author={Hug, LA and Baker, BJ and Anantharaman, K and Brown, CT and Probst, AJ and Castelle, CJ and Banfield, JF},
     year={2016},
     institution={Retrieved 2021-11-04, from http://www. nature. com/articles/nmicrobiol201648~…}
   }
   
   @article{zitnik2019evolution,
     title={Evolution of resilience in protein interactomes across the tree of life},
     author={Zitnik, Marinka and Sosi{\v{c}}, Rok and Feldman, Marcus W and Leskovec, Jure},
     journal={Proceedings of the National Academy of Sciences},
     volume={116},
     number={10},
     pages={4426--4433},
     year={2019},
     publisher={National Acad Sciences}
   }


-----

----------
sentiment
----------

Graphs in `sentiment <http://help.sentiment140.com/for-students>`_ are parsed dependency tree from tweets. Specifically, we used SpaCy
library to parse the dependency trees of tokens and obtain the node embeddings of the trees. The
binary class of each graph indicates whether the sentiment in tweet is positive or negative. For
Time-IL, we formulate 11 tasks according to the timestamps of the tweets. Specifically, we constructed the tasks with the tweets posted in the same day.

Statistics:

- Number of Graphs: 5,500
- Average Number of Nodes: 13.43
- Average Number of Edges: 23.71
- Number of Node Features: 300
- Number of Edge Features: 0
- Number of Classes: 2
- Supported Incremental Settings:
   
   + Time-IL with 11 tasks

Citing:

.. code-block::

   @article{go2009twitter,
     title={Twitter sentiment classification using distant supervision},
     author={Go, Alec and Bhayani, Richa and Huang, Lei},
     journal={CS224N project report, Stanford},
     volume={1},
     number={12},
     pages={2009},
     year={2009}
   }


