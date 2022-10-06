Datasets for Link-Level Problems
===================================

----------------------------------
BitCoin-OTC (Link Classification)
----------------------------------

The dataset `BitCoin-OTC <https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html>`_ is a who-trust-whom network, where nodes are users
of a bitcoin-trading platform. Each directed edge has an integer rating between −10 to 10 and a
timestamp. The ratings are divided into 6 groups. Two of them are used separately for Task-IL
and accumulated for Class-IL. For Time-IL, we formulate 7 tasks using the timestamps, where the
signs of the ratings are used as binary classes. Since there is no external node feature, we use in-
and out-degrees as node features.

Statistics:

- Nodes: 5,881
- Edges: 35,592
- Number of Classes: 21 
- Supported Incremental Settings:
   
   + Task-IL with 3 tasks
   + Class-IL with 3 tasks
   + Time-IL with 7 tasks

------------------------------
Wiki-CS (Link Prediction)
------------------------------

`Wiki-CS <https://github.com/pmernyei/wiki-cs-dataset>`_ is a hyperlink network between computer science articles.
Each article has a label indicating one of the 10 subfields that it belongs to. For Domain-IL, the
node labels are used as domains, and specifically, the edges are divide into 10 groups, according
to the labels of their endpoints. If the domains of its two endpoints are different, the domain
considered in a later task is assigned to the edge.

Statistics:

- Nodes: 11,701
- Edges: 431,726
- Number of Classes: 2
- Number of Node Features: 300
- Supported Incremental Settings:
   
   + Domain-IL with 10 tasks

------------------------------
ogbl-collab (Link Prediction)
------------------------------

`ogbl-collab <https://ogb.stanford.edu/docs/linkprop/#ogbl-collab>`_ s a co-authorship network, where nodes are
authors. We use publication years to form 9 groups for the Time-IL setting.

Statistics:

- Nodes: 235,868
- Edges: 1,285,465
- Number of Classes: 2
- Number of Node Features: 128
- Supported Incremental Settings:
   
   + Time-IL with 9 tasks