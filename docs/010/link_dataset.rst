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

```bibtex
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
```

------------------------------
Wiki-CS (Link Prediction)
------------------------------

`Wiki-CS <https://github.com/pmernyei/wiki-cs-dataset>`_ is a hyperlink network between computer science articles.
Each article has a label indicating one of the 10 subfields that it belongs to. For Domain-IL, the
node labels are used as domains, and specifically, the edges are divide into 54 groups, according
to the labels of their endpoints.

Statistics:

- Nodes: 11,701
- Edges: 431,726
- Number of Classes: 2
- Number of Node Features: 300
- Supported Incremental Settings:
   
   + Domain-IL with 54 tasks

------------------------------
ogbl-collab (Link Prediction)
------------------------------

`ogbl-collab <https://ogb.stanford.edu/docs/linkprop/#ogbl-collab>`_ s a co-authorship network, where nodes are
authors. We use publication years to form 50 groups for the Time-IL setting. Due to the imbalance on the publication years of the considered papers, we constructed formed the first task using the paper published before 1971. From the second task, we constructed the `i`-th task with the paper published in `1969 + i`.
Each node feature is obtained by averaging the word embeddings of published papers.

Statistics:

- Nodes: 235,868
- Edges: 1,285,465
- Number of Classes: 2
- Number of Node Features: 128
- Supported Incremental Settings:
   
   + Time-IL with 50 tasks

------------------------------
Facebook (Link Prediction)
------------------------------

`Facebook <https://github.com/benedekrozemberczki/datasets#facebook-page-page-networks>`_ is a social network, where nodes are pages of Facebook, and its class belongs to one
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

```bibtex
@inproceedings{rozemberczki2019gemsec,    
                title={GEMSEC: Graph Embedding with Self Clustering},    
                author={Rozemberczki, Benedek and Davies, Ryan and Sarkar, Rik and Sutton, Charles},    
                booktitle={ASONAM},    
                year={2019},    
}
```

------------------------------
Ask-Ubuntu (Link Prediction)
------------------------------

Nodes in `Ask-Ubuntu <http://snap.stanford.edu/data/sx-askubuntu.html>`_ are users of askubuntu, and edges indicate there is interaction between the users. The edges are divided into 69 groups according to the timestamps for Time-IL. Specifically, we constructed the tasks with the papers published in the same month. We make the graph to be undirected, and since there is no external node feature, we use degrees as node features.

Statistics:

- Nodes: 159,313
- Edges: 507,988
- Number of Classes: 2
- Number of Node Features: 1
- Supported Incremental Settings:
   
   + Time-IL with 69 tasks

Citing:

```bibtex
@inproceedings{paranjape2017motifs,
  title={Motifs in temporal networks},
  author={Paranjape, Ashwin and Benson, Austin R and Leskovec, Jure},
  booktitle={Proceedings of the tenth ACM international conference on web search and data mining},
  pages={601--610},
  year={2017}
}

------------------------------
Gowalla (Link Prediction)
------------------------------

`Gowalla <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/gowalla>`_ consists of check-in history from a location-based social networking platform where users share their locations through check-ins. Each node represents either a user or a location, and each edge represents a user’s check-in at a location. For Time-IL, we organize 10 tasks chronologically based on check-in timestamps. This scenario and the next one (i.e., MovieLens) are directly related to personalized recommendation systems (spec., movie and POI recommendations), which are essential for helping users find relevant options among numerous candidates.

Statistics:

- Nodes: 70,839
- Edges: 1,027,370
- Number of Classes: 2
- Number of Node Features: 2
- Supported Incremental Settings:
   
   + Time-IL with 10 tasks

Citing:

```bibtex
@inproceedings{liang2016modeling,
  title={Modeling user exposure in recommendation},
  author={Liang, Dawen and Charlin, Laurent and McInerney, James and Blei, David M},
  booktitle={Proceedings of the 25th international conference on World Wide Web},
  pages={951--961},
  doi = {10.48550/arXiv.1510.07025},
  year={2016}
}

@inproceedings{wang2019neural,
  title={Neural graph collaborative filtering},
  author={Wang, Xiang and He, Xiangnan and Wang, Meng and Feng, Fuli and Chua, Tat-Seng},
  booktitle={Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval},
  pages={165--174},
  year={2019}
}
```

------------------------------
MovieLens (Link Prediction)
------------------------------

`MovieLens <https://grouplens.org/datasets/movielens/1m/>`_ is a movie-rating dataset. We convert it into a graph where nodes represent either users or movies. An edge is created between a user and a movie if and only if the user gives the movie a rating of 4 or higher. For Time-IL, we organize 10 tasks chronologically based on the rating timestamps. This scenario and the previous one (i.e., Gowalla) are directly related to personalized recommendation systems (spec., movie and POI recommendations), which are essential for helping users find relevant options among numerous candidates.

Statistics:

- Nodes: 9,992
- Edges: 575,281
- Number of Classes: 2
- Number of Node Features: 42
- Supported Incremental Settings:
   
   + Time-IL with 10 tasks

Citing:

```bibtex
@article{harper2015movielens,
  title={The movielens datasets: History and context},
  author={Harper, F Maxwell and Konstan, Joseph A},
  journal={Acm transactions on interactive intelligent systems (tiis)},
  volume={5},
  number={4},
  pages={1--19},
  year={2015},
  doi = {10.1145/2827872},
  publisher={Acm New York, NY, USA}
}
```