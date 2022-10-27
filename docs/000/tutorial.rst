Tutorial
===================================

BeGin is a framework containing the following core components:
- ScenarioLoader: This module provides built-in continual learning scenarios to evaluate the performances of graph continual learning methods.
- Evaluator: This module provides the evaluator, which computes basic metrics based on the ground-truth and predicted answers.
- Trainer: This module manages the overall training procedure of user-defined continual learning algorithms, including preparing the dataloader, training, and validation, so that users only have to implement novel parts of their methods.

In this material, we briefly describe how to perform graph continual learning with those components using some examples.

Step 1. ScenarioLoader and Evaluation Metric
--------

In order to evaluate graph CL methods, we need to prepare (1) graph datasets with multi-class, domain, or timestamps, (2) incremental settings, and (3) proper evaluation metric for the settings. To reduce such efforts, BeGin provides various benchmark scenarios based on graph-related problems and incremental settings for continual learning, and built-in evaluation metrics. For example, using BeGin, user can load the task-incremental node classification scenario on cora dataset in just one line of code.

.. code-block::

  from begin.scenarios import NodeClassificationIL
  NodeClassificationIL(dataset_name='cora', num_tasks=3, metric='accuracy', save_path='/data', incr_type='task')

Currently, BeGin supports 19 Node Classification (NC), Link Classification (LC), Link Prediction (LP), Graph Classification (GC) scenarios with the following incremental settings for continual learning with graph data. (For the provided scenarios with real-world datasets and evaluation metrics, see :ref:`AAA` and :ref:`BBB`, respectively.)

- Task-incremental (Task-IL): In this incremental setting, the set of classes consisting each task varies with tasks, and they are often disjoint. In addition, for each query at evaluation, the corresponding task information is provided, and thus its answer is predicted among the classes considered in the task. This setting is applied to NC and GC tasks, where the sets of classes can vary with tasks, and for NC and LC tasks, the input graph is fixed.

- Class-incremental (Class-IL): In this incremental setting, the set of classes grows over tasks. In addition, for each query at evaluation, the corresponding task is NOT provided, and thus its answer is predicted among all classes seen so far. This setting is applied to NC and GC tasks, where the sets of classes can vary with tasks, and for NC and LC tasks, the input graph is fixed.

- Domain-incremental (Domain-IL): In this incremental setting, we divided entities (i.e., nodes, edges, and graphs) over tasks according to their domains, which are additionally given. For NC, the nodes of the input graph are divided into NC tasks according to their domains. For LC and LP, the links of the input graph and the input queries are divided according to their domains, respectively. For GC, the links of the input graph are divided into LC tasks according to their domains. For NC, LC, and LP tasks, the input graph is fixed.

- Time-incremental (Time-IL): In this incremental setting except for GC, we consider a dynamic graph evolving over time, and the set of classes may or may not vary across tasks. For NC, LC, and LP, the input graph of i-th task is the i-th snapshot of the dynamic graph. For GC, the snapshots of the dynamic graph are grouped and assigned to tasks in chronological order.


Step 2. Trainer
--------

For usability, BeGin provides the trainer, which users can extend when implementing and benchmarking new methods. It manages the overall training procedure, including preparing the dataloader, training, and validation, so that users only have to implement novel parts of their methods. As in Avalanche, the trainer divides the training procedure of continual learning as a series of events. For example, the subprocesses in the training procedure where the trainer (a) receives the input for the current task, (b) trains the model for one iteration for the current task, and (c) handles the necessary tasks before and after the training. as events. Each event is modularized as a function, which users can fill out, and the trainer proceeds the training procedure with the event functions.

Currently, BeGin supports the following event functions. Note that implementing each event function is optional. If the user-defined functions is not provided, Trainer performs training and evaluation with the corresponding basic pre-implemented operations. See :ref:`CCC` for the detailed arguments and role of the event functions.

- :func:`initTraining`: This function is called only once, when the training procedure begins. 
- :func:`prepareLoader': This function is called once for each task when generating dataloaders for train/validation/test.
- :func:`processBeforeTraining': This function is called once for each task, right after .
- :func:`processTrainIteration`: This function is called once for every training iteration.
- :func:`processEvalIteration`: This function is called once for every training iteration.



