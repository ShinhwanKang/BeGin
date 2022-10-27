Tutorial
===================================


In this material, we briefly describe how to perform graph continual learning with BeGin using some examples.

Step 0. Prepare Continual Learning Scenario
--------

BeGin provides various benchmark scenarios based on graph-related problems and incremental settings for continual learning. Currently, BeGin supports Node Classification (NC), Link Classification (LC), Link Prediction (LP), Graph Classification (GC) scenarios with the following incremental settings for continual learning with graph data.

- Task-incremental (Task-IL): In this incremental setting, the set of classes consisting each task varies with tasks, and they are often disjoint. In addition, for each query at evaluation, the corresponding task information is provided, and thus its answer is predicted among the classes considered in the task. This setting is applied to NC and GC tasks, where the sets of classes can vary with tasks, and for NC and LC tasks, the input graph is fixed.

- Class-incremental (Class-IL): In this incremental setting, the set of classes grows over tasks. In addition, for each query at evaluation, the corresponding task is NOT provided, and thus its answer is predicted among all classes seen so far. This setting is applied to NC and GC tasks, where the sets of classes can vary with tasks, and for NC and LC tasks, the input graph is fixed.

- Domain-incremental (Domain-IL): In this incremental setting, we divided entities (i.e., nodes, edges, and graphs) over tasks according to their domains, which are additionally given. For NC, the nodes of the input graph are divided into NC tasks according to their domains. For LC and LP, the links of the input graph and the input queries are divided according to their domains, respectively. For GC, the links of the input graph are divided into LC tasks according to their domains. For NC, LC, and LP tasks, the input graph is fixed.

- Time-incremental (Time-IL): In this incremental setting except for GC, we consider a dynamic graph evolving over time, and the set of classes may or may not vary across tasks. For NC, LC, and LP, the input graph of i-th task is the i-th snapshot of the dynamic graph. For GC, the snapshots of the dynamic graph are grouped and assigned to tasks in chronological order.
