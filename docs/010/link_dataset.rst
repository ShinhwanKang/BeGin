Datasets for Link-Level Problems
===================================

----------------------------------
Bitcoin-OTC (Link Classification)
----------------------------------

The dataset `Bitcoin-OTC <https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html>`_ is a who-trust-whom network, where nodes are users
of a bitcoin-trading platform. Each directed edge has an integer rating between −10 to 10 and a
timestamp. The ratings are divided into 6 groups. Two of them are used separately for Task-IL
and accumulated for Class-IL. For Time-IL, we formulate 7 tasks using the timestamps, where the
signs of the ratings are used as binary classes. Since there is no external node feature, we use in-degrees
and out-degrees as node features.

Statistics:

- Nodes: 5,881
- Edges: 35,592
- Number of Classes: 21 
- Supported Incremental Settings:
   
   + Task-IL with 3 tasks
   + Class-IL with 3 tasks
   + Time-IL with 7 tasks

Citing:

.. code-block::

   @inproceedings{kumar2016edge,
     title={Edge weight prediction in weighted signed networks},
     author={Kumar, Srijan and Spezzano, Francesca and Subrahmanian, VS and Faloutsos, Christos},
     booktitle={ICDM},
     year={2016},
   }
   
   @inproceedings{kumar2018rev2,
     title={Rev2: Fraudulent user prediction in rating platforms},
     author={Kumar, Srijan and Hooi, Bryan and Makhija, Disha and Kumar, Mohit and Faloutsos, Christos and Subrahmanian, VS},
     booktitle={WSDM},
     year={2018},
   }


------------------------------
Wiki-CS (Link Prediction)
------------------------------

`Wiki-CS <https://github.com/pmernyei/wiki-cs-dataset>`_ is a hyperlink network between computer science articles.
Each article has a label indicating one of the 10 subfields that it belongs to. For Domain-IL, the
node labels are used as domains, and specifically, the edges are divide into 54 groups, according
to the labels of their endpoints. According to the original paper, the node features are computed as the average of pretrained GloVe embeddings.

Statistics:

- Nodes: 11,701
- Edges: 431,726
- Number of Classes: 2
- Number of Node Features: 300
- Supported Incremental Settings:
   
   + Domain-IL with 54 tasks

Citing:

.. code-block::

   @article{mernyei2020wiki,
     title={Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks},
     author={Mernyei, P{\'e}ter and Cangea, C{\u{a}}t{\u{a}}lina},
     journal={arXiv preprint arXiv:2007.02901},
     year={2020}
   }

------------------------------
ogbl-collab (Link Prediction)
------------------------------

`ogbl-collab <https://ogb.stanford.edu/docs/linkprop/#ogbl-collab>`_ s a co-authorship network, where nodes are
authors. We use publication years to form 50 groups for the Time-IL setting. Due to the imbalance in the publication years of the considered papers, we constructed the first task using the paper published before the year `1971`. For each subsequent `i`-th task, we used the papers published in the year `(1969 + i)`.

Statistics:

- Nodes: 235,868
- Edges: 1,285,465
- Number of Classes: 2
- Number of Node Features: 128
- Supported Incremental Settings:
   
   + Time-IL with 50 tasks

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

------------------------------
facebook (Link Prediction)
------------------------------

`facebook <https://github.com/benedekrozemberczki/datasets#facebook-page-page-networks>`_ is a social network, where nodes are pages of Facebook, and its class belongs to one
among 8 categories. Edges indicate mutual likes among the pages, and they exist only between
pages within the same category. We divide the edges into 8 groups, according to the labels of their
endpoints. We make the graph to be undirected, and since there is no external node feature, we use degrees as node features.

Statistics:

- Nodes: 134,833
- Edges: 1,380,293
- Number of Classes: 2
- Number of Node Features: 1
- Supported Incremental Settings:
   
   + Domain-IL with 8 tasks

Citing:

.. code-block::

   @inproceedings{rozemberczki2019gemsec,    
                   title={GEMSEC: Graph Embedding with Self Clustering},    
                   author={Rozemberczki, Benedek and Davies, Ryan and Sarkar, Rik and Sutton, Charles},    
                   booktitle={ASONAM},    
                   year={2019},    
   }


------------------------------
askubuntu (Link Prediction)
------------------------------

Nodes in `askubuntu <http://snap.stanford.edu/data/sx-askubuntu.html>`_ are users of askubuntu, and edges indicate there is interaction between the
users. The edges are divided into 69 groups according to the timestamps for Time-IL. Specifically, we used the interactions occurring within the same month to form each task. We make the graph to be undirected, and since there is no external node feature, we use degrees as node features.

Statistics:

- Nodes: 159,313
- Edges: 507,988
- Number of Classes: 2
- Number of Node Features: 1
- Supported Incremental Settings:
   
   + Time-IL with 69 tasks

Citing:

.. code-block::

   @inproceedings{paranjape2017motifs,
     title={Motifs in temporal networks},
     author={Paranjape, Ashwin and Benson, Austin R and Leskovec, Jure},
     booktitle={Proceedings of the tenth ACM international conference on web search and data mining},
     pages={601--610},
     year={2017}
   }
