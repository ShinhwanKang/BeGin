===================================
Tutorial
===================================

BeGin is a framework containing the following core components:

- ScenarioLoader: This module provides built-in continual learning scenarios to evaluate the performances of graph continual learning methods.
- Evaluator: This module provides the evaluator, which computes basic metrics based on the ground-truth and predicted answers.
- Trainer: This module manages the overall training procedure of user-defined continual learning algorithms, including preparing the dataloader, training, and validation, so that users only have to implement novel parts of their methods.

In this material, we briefly describe how to perform graph continual learning with those components using some examples.

------------------------
ScenarioLoader and Evaluation Metric
------------------------

In order to evaluate graph CL methods, we need to prepare (1) graph datasets with multi-class, domain, or timestamps, (2) incremental settings, and (3) proper evaluation metric for the settings. To reduce such efforts, BeGin provides various benchmark scenarios based on graph-related problems and incremental settings for continual learning, and built-in evaluation metrics. For example, using BeGin, user can load the task-incremental node classification scenario on cora dataset in just one line of code.

.. code-block::

  from begin.scenarios import NodeClassificationScenarioLoader
  NodeClassificationScenarioLoader(dataset_name='cora', num_tasks=3, metric='accuracy', save_path='/data', incr_type='task')

Currently, BeGin supports 19 Node Classification (NC), Link Classification (LC), Link Prediction (LP), Graph Classification (GC) scenarios with the following incremental settings for continual learning with graph data. (For the provided scenarios with real-world datasets and evaluation metrics, see :ref:`AAA` and :ref:`BBB`, respectively.)

- Task-incremental (Task-IL): In this incremental setting, the set of classes consisting each task varies with tasks, and they are often disjoint. In addition, for each query at evaluation, the corresponding task information is provided, and thus its answer is predicted among the classes considered in the task. This setting is applied to NC and GC tasks, where the sets of classes can vary with tasks, and for NC and LC tasks, the input graph is fixed.

- Class-incremental (Class-IL): In this incremental setting, the set of classes grows over tasks. In addition, for each query at evaluation, the corresponding task is NOT provided, and thus its answer is predicted among all classes seen so far. This setting is applied to NC and GC tasks, where the sets of classes can vary with tasks, and for NC and LC tasks, the input graph is fixed.

- Domain-incremental (Domain-IL): In this incremental setting, we divided entities (i.e., nodes, edges, and graphs) over tasks according to their domains, which are additionally given. For NC, the nodes of the input graph are divided into NC tasks according to their domains. For LC and LP, the links of the input graph and the input queries are divided according to their domains, respectively. For GC, the links of the input graph are divided into LC tasks according to their domains. For NC, LC, and LP tasks, the input graph is fixed.

- Time-incremental (Time-IL): In this incremental setting except for GC, we consider a dynamic graph evolving over time, and the set of classes may or may not vary across tasks. For NC, LC, and LP, the input graph of i-th task is the i-th snapshot of the dynamic graph. For GC, the snapshots of the dynamic graph are grouped and assigned to tasks in chronological order.

--------
Trainer
--------

For usability, BeGin provides the trainer, which users can extend when implementing and benchmarking new methods. It manages the overall training procedure, including preparing the dataloader, training, and validation, so that users only have to implement novel parts of their methods. As in Avalanche, the trainer divides the training procedure of continual learning as a series of events. For example, the subprocesses in the training procedure where the trainer (a) receives the input for the current task, (b) trains the model for one iteration for the current task, and (c) handles the necessary tasks before and after the training. as events. Each event is modularized as a function, which users can fill out, and the trainer proceeds the training procedure with the event functions.

Currently, BeGin supports the following event functions. Note that implementing each event function is optional. If the user-defined functions is not provided, Trainer performs training and evaluation with the corresponding basic pre-implemented operations. See :ref:`CCC` for the detailed arguments and role of the event functions.

- :func:`initTraining`: This function is called only once, when the training procedure begins. 
- :func:`prepareLoader': This function is called once for each task when generating dataloaders for train/validation/test. Given dataset for each task, it should return dataloaders for training, validation, and test.
- :func:`processBeforeTraining': This function is called once for each task, right after the :func:`prepareLoader' event function.
- :func:`processTrainIteration`: This function is called for every training iteration. When the current batched inputs, model, and optimizer are given, it should perform single training iteration and return the information or outcome during the iteration.  
- :func:`processEvalIteration`: This function is called for every evaluation iteration. When the current batched inputs and trained model are given, it should perform single evaluation iteration and return the information or outcome during the iteration.
- :func:`inference`: This function is called for every inference step in the training procedure. 
- :func:`beforeInference`: This function is called right after the :func:`_model_inference`.
- :func:`afterInference`: This function is called right after the :func:`_model_inference`.
- :func:`reduceTrainingStats`: This function is called at the end of every training step. Given the returned values of the :func:`processTrainIteration` event function, it should returns overall and reduced statistics of the current training step.
- :func:`reduceEvalStats`: This function is called at the end of every evaluation step. Given the returned values of the :func:`processEvalIteration` event function, it should returns overall and reduced statistics of the current evaluation step.
- :func:`processTrainingLogs`: This function is called right after the :func:`reduceTrainingStats` event function. It should generates training logs for the current training iteration.
- :func:`procssAfterEachIteration`: This function is called at the end of the training iteration. When the outcome from :func:`reduceTrainingStats` and :func:`reduceEvalStats` are given, it should determine whether the trainer should stop training for the current task or not.

Suppose we need to implement Elastic Weight Consolidation (EWC) algorithm for class-IL node classification using BeGin. EWC algorithm is a regularization-based CL algorithm for generic data. Specifically, it uses weighted L2 penalty term which is determined by the learned weights from the previous tasks as in the following equation:

.. math::

\mathcal{L}(\theta) = \mathcal{L}_i(\theta) + \sum_{j=1}^{i-1} \frac{\lambda}{2} F_j (\theta - \theta^*_j)^2,

where :math:`\theta` is current weights of the model, :math:`\theta^*_j` is learned weights until the :math:`j`-th task, :math:`\lambda > 0` is a hyperparameter, and :math:`F_j` is the diagonal part of the Fisher information matrix until the :math:`j`-th task computed as square of the first derivatives.


Step 1. Extending the base 
=============

BeGin provides basic implementation of trainer for each graph-related problem. Each basic trainer follows the incremental learning schemes, but no CL technique is applied. For example, if we want to implement CL algorithm for NC task, you need to extend :func:`NCTrainer` to reduce your efforts for implementing user-defined functions on managing the overall procedure.

.. code-block::

  from begin.trainers import NCTrainer
  class EWCTaskILNCTrainer(NCTrainer):
      pass

Step 2. Setting initial states for the algorithm (:func:`initTraining`)
=============

As in the aformentioned equation, EWC algorithm requires to store learned weights and Fisher information matrices from the previous tasks to compute the regualarization term. However, but they cannot be obtained on the current task. In order to resolve this issue, the trainer provides a dictionary called :func:`training_states` which can store intermediate results and be shared by events as the parameter of the event functions. To set the initial states, BeGin provides :func:`initTraining` event function, and the trainer set the initial states to the returned dictionary from the event function. In this example, we assigned :func:`fishers` to store the fisher information matrices of each task and :func:`params` to store the learned weights of each task, as shown in the code below.

.. code-block::

  from begin.trainers import NCTrainer
  class EWCTaskILNCTrainer(NCTrainer):
      def initTraining(self, model, optimizer):
          return {'fishers': [], 'params': []}
      
Step 3. Storing previous weights and Fisher matrix (:func:`processAfterTraining`)
=============
  from begin.trainers import NCTrainer
  class EWCTaskILNCTrainer(NCTrainer):
      def initTraining(self, model, optimizer):
          return {'fishers': [], 'params': []}
          
      def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
          super()._processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
          params = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
          fishers = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
          train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
          total_num_items = 0
          for i, _curr_batch in enumerate(iter(train_loader)):
              curr_model.zero_grad()
              curr_results = self._model_inference(curr_model, _curr_batch, curr_training_states)
              curr_results['loss'].backward()
              curr_num_items =_curr_batch[1].shape[0]
              total_num_items += curr_num_items
              for name, p in curr_model.named_parameters():
                  params[name] = p.data.clone().detach()
                  fishers[name] += (p.grad.data.clone().detach() ** 2) * curr_num_items
                    
          for name, p in curr_model.named_parameters():
              fishers[name] /= total_num_items
                
          curr_training_states['fishers'].append(fishers)
          curr_training_states['params'].append(params)
          
Step 4. Storing previous weights and Fisher matrix (:func:`processTrainIteration` and :func:`after_inference`)
=============

--------
Combining ScenarioLoader, Evaluator, Trainer
--------
