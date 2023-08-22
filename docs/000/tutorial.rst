===================================
Tutorial
===================================

BeGin is a framework containing the following core components:

- ScenarioLoader: This module provides built-in continual learning scenarios to evaluate the performances of graph continual learning methods.
- Evaluator: This module provides the evaluator, which computes basic metrics based on the ground-truth and predicted answers.
- Trainer: This module manages the overall training procedure of user-defined continual learning algorithms, including preparing the dataloader, training, and validation, so that users only have to implement novel parts of their methods.

In this material, we briefly describe how to perform graph continual learning with those components using some examples.

----------------------------------------
ScenarioLoader and Evaluation Metric
----------------------------------------

In order to evaluate graph CL methods, we need to prepare (1) graph datasets with multi-class, domain, or timestamps, (2) incremental settings, and (3) proper evaluation metric for the settings. To reduce such efforts, BeGin provides various benchmark scenarios based on graph-related problems and incremental settings for continual learning, and built-in evaluation metrics. For example, using BeGin, users can load the class-incremental node classification scenario on ogbn-arxiv dataset in just one line of code.

.. code-block:: python

  >>> from begin.scenarios.nodes import NCScenarioLoader
  >>> NCScenarioLoader(dataset_name='ogbn-arxiv', num_tasks=8, metric='accuracy', save_path='/data', incr_type='class')

Currently, BeGin supports 19 Node Classification (NC), Link Classification (LC), Link Prediction (LP), Graph Classification (GC) scenarios with the following incremental settings for continual learning with graph data.

- Task-incremental (Task-IL): In this incremental setting, the set of classes consisting each task varies with tasks, and they are often disjoint. In addition, for each query at evaluation, the corresponding task information is provided, and thus its answer is predicted among the classes considered in the task. This setting is applied to NC and GC tasks, where the sets of classes can vary with tasks, and for NC and LC tasks, the input graph is fixed.

- Class-incremental (Class-IL): In this incremental setting, the set of classes grows over tasks. In addition, for each query at evaluation, the corresponding task is NOT provided, and thus its answer is predicted among all classes seen so far. This setting is applied to NC and GC tasks, where the sets of classes can vary with tasks, and for NC and LC tasks, the input graph is fixed.

- Domain-incremental (Domain-IL): In this incremental setting, we divided entities (i.e., nodes, edges, and graphs) over tasks according to their domains, which are additionally given. For NC, the nodes of the input graph are divided into NC tasks according to their domains. For LC and LP, the links of the input graph and the input queries are divided according to their domains, respectively. For GC, the links of the input graph are divided into LC tasks according to their domains. For NC, LC, and LP tasks, the input graph is fixed.

- Time-incremental (Time-IL): In this incremental setting except for GC, we consider a dynamic graph evolving over time, and the set of classes may or may not vary across tasks. For NC, LC, and LP, the input graph of i-th task is the i-th snapshot of the dynamic graph. For GC, the snapshots of the dynamic graph are grouped and assigned to tasks in chronological order.

--------
Trainer
--------

For usability, BeGin provides the trainer, which users can extend when implementing and benchmarking new methods. It manages the overall training procedure, including preparing the dataloader, training, and validation, so that users only have to implement novel parts of their methods. As in Avalanche, the trainer divides the training procedure of continual learning as a series of events. For example, the subprocesses in the training procedure where the trainer (a) receives the input for the current task, (b) trains the model for one iteration for the current task, and (c) handles the necessary tasks before and after the training. as events. Each event is modularized as a function, which users can fill out, and the trainer proceeds the training procedure with the event functions.

Currently, BeGin supports the following event functions. Note that implementing each event function is optional. If a user-defined event function is not provided, the trainer performs training and evaluation with the corresponding basic pre-implemented operations. Thus, users do not need to implement the whole training and evaluation procedure, but only the necessary parts. See `here <../040/common.html>`_ for the detailed arguments and roles of the event functions. Currently, users can override the following built-in event functions: 

- :func:`initTrainingStates`: This function is called only once, when the training procedure begins.
- :func:`prepareLoader`: This function is called once for each task when generating dataloaders for training, validation, and test. Given the dataset for each task, it should return dataloaders for training, validation, and test.
- :func:`processBeforeTraining`: This function is called once for each task, right after the :func:`prepareLoader` event function terminates.
- :func:`processTrainIteration`: This function is called for every training iteration. When the current batched inputs, model, and optimizer are given, it should perform a single training iteration and return the information or outcome during the iteration. 
- :func:`processEvalIteration`: This function is called for every evaluation iteration. When the current batched inputs and trained model are given, it should perform a single evaluation iteration and return the information or outcome during the iteration.
- :func:`inference`: This function is called for every inference step in the training procedure.
- :func:`beforeInference`: This function is called right before the :func:`inference` begins.
- :func:`afterInference`: This function is called right after the :func:`inference` terminates.
- :func:`_reduceTrainingStats`: This function is called at the end of every training step. Given the returned values of the :func:`processTrainIteration` event function, it should return overall and reduced statistics of the current training step.
- :func:`_reduceEvalStats`: This function is called at the end of every evaluation step. Given the returned values of the :func:`processEvalIteration` event function, it should return overall and reduced statistics of the current evaluation step.
- :func:`processTrainingLogs`: This function is called right after the :func:`reduceTrainingStats` event function terminates. It should generate training logs for the current training iteration.
- :func:`procssAfterEachIteration`: This function is called at the end of the training iteration. When the outcome from :func:`reduceTrainingStats` and :func:`reduceEvalStats` are given, it should determine whether the trainer should stop training for the current task or not.
- :func:`processAfterTraining`: This function is called once for each task when the trainer completes training for the current task.

Suppose we implement Elastic Weight Consolidation (EWC) algorithm for class-IL node classification using BeGin. EWC algorithm is a regularization-based CL algorithm for generic data. Specifically, it uses weighted L2 penalty term which is determined by the learned weights from the previous tasks as in the following equation:

.. math:: \mathcal{L}(\theta) = \mathcal{L}_i(\theta) + \sum_{j=1}^{i-1} \frac{\lambda}{2} F_j (\theta - \theta^*_j)^2,

where :math:`\theta` is current weights of the model, :math:`\theta^*_j` is learned weights until the :math:`j`-th task, :math:`\lambda > 0` is a hyperparameter, and :math:`F_j` is the diagonal part of the Fisher information matrix until the :math:`j`-th task computed as square of the first derivatives.


Step 1. Extending the base 
============================

BeGin provides base trainer class that makes the training behavior exactly the same as the Bare model. Based on the base class, users can implement their CL algorithm by extending the class and substituting some default event functions with user-defined ones.

.. code-block:: python

  from begin.trainers.nodes import NCTrainer
  class NCClassILEWCTrainer(NCTrainer):
      pass

Step 2. Setting initial states for the algorithm (:func:`initTrainingStates`)
===============================================================================

As stated in the aformentioned equation, EWC requires storing the learned weights and Fisher information matrices from the previous tasks to compute the regularization term. However, they cannot be obtained on the current task. In order to resolve this issue, the trainer provides a dictionary called `training_states`. The dictionary can be used to store intermediate results and can be shared by events in the form of an argument (i.e., input parameter) of the event functions. To set the initial states, the user can extend the base trainer with their modified :func:`initTrainingStates` event function, which initializes the states for running EWC.

.. code-block:: python

  from begin.trainers.nodes import NCTrainer
  class NCClassILEWCTrainer(NCTrainer):
      def initTrainingStates(self, model, optimizer):
          return {'fishers': [], 'params': []}
      
Step 3. Storing previous weights and Fisher matrix (:func:`processAfterTraining`)
====================================================================================

In order to compute the penalty term at task :math:`i`, we need the learned weights :math:`\theta^*_j` and Fisher information matrix :math:`F_j` for every task :math:`j < i`. Hence, we need to store them at the end of each task, and this can naturally be implemented in the event function :func:`processAfterTraining`, which is called at the end of each task. In the example below, `curr_training_states['params'][j-1]` and `curr_training_states['fishers'][j-1]` store the learned weights and the Fisher information matrix of task :math:`j`, respectively.

.. code-block:: python

  from begin.trainers.nodes import NCTrainer
  class NCClassILEWCTrainer(NCTrainer):
      def initTrainingStates(self, model, optimizer):
          return {'fishers': [], 'params': []}
          
      def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
          super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
          params = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
          fishers = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
          train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
          total_num_items = 0
          for i, _curr_batch in enumerate(iter(train_loader)):
              curr_model.zero_grad()
              curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
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
          
Step 4. Computing penalty term and Performing regularization (:func:`processTrainIteration` and :func:`afterInference`)
========================================================================================================================

The penalty term in the above equation is used for regularization during a backpropagation process. The computation of the term should be performed at the end of training for every task, and thus it is implemented in :func:`afterInference`.
In the event function, the argument (i.e., input parameter) `curr_training_states` contains the Fisher information matrices and the previously learned weights based on which the penalty term `loss_reg` is computed.
The event function also has the argument `results`, which contains the prediction result and loss of the current model computed in the :func:`inference` function. 
Thus, the overall loss including the penalty term can be obtained by summing up `results['loss']` and `loss_reg`.

.. code-block:: python

  from begin.trainers.nodes import NCTrainer
  class NCClassILEWCTrainer(NCTrainer):
      def initTrainingStates(self, model, optimizer):
          return {'fishers': [], 'params': []}
          
      def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
          super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
          params = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
          fishers = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
          train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
          total_num_items = 0
          for i, _curr_batch in enumerate(iter(train_loader)):
              curr_model.zero_grad()
              curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
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
      
      def afterInference(self, results, model, optimizer, _curr_batch, training_states):
          loss_reg = 0
          for _param, _fisher in zip(training_states['params'], training_states['fishers']):
              for name, p in model.named_parameters():
                  l = self.lamb * _fisher[name]
                  l = l * ((p - _param[name]) ** 2)
                  loss_reg = loss_reg + l.sum()
          total_loss = results['loss'] + loss_reg
          total_loss.backward()
          optimizer.step()
          return {'loss': total_loss.item(),
                  'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}

The above code presents the complete implementation of EWC for node classification under Class-IL. Similarly, various CL algorithms can be developed by modifying specific event functions, without the need to manage the overall training and evaluation procedures. Refer to `here <../040/common.html>`_ for the detailed explanation of the event functions and their parameters.

------------------------------------------------
Combining ScenarioLoader, Evaluator, Trainer
------------------------------------------------

So far we have learned how to load each component of BeGin. The last step is to combine the components to perform the experiments under the prepared scenario and trainer, and this process also takes just a few lines of code.

.. code-block:: python

  from begin.scenarios.nodes import NCScenarioLoader
  
  scenario = NCScenarioLoader(dataset_name='ogbn-arxiv', num_tasks=8, metric='accuracy', save_path='./data', incr_type='class')
  benchmark = NCClassILEWCTrainer(model = GCN(scenario.num_feats, scenario.num_classes, 256, dropout=0.25),
                                  scenario = scenario,
                                  optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3),
                                  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1),
                                  device = torch.device('cuda:0'),
                                  scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=args.lr * 0.001 * 2., verbose=True))
  results = benchmark.run(epoch_per_task = 1000)
  
To run the experiment, trainer object in BeGin requires a learnable model, a CL scenraio, a proper loss function to train the model, a function to generate optimizer and scheduler, and the other auxilary arguments to customize the trainer. After creating the object, users can start the experiment by calling the member function `results` of the trainer object.

In BeGin, at the end of each task, the trainer measures the performance of all tasks. When the procedure is completed, the trainer returns the evaluation results, which is in the form of a matrix. In the matrix, the (i,j)-th entry contains the performance evaluated using the test data of task j when the training of task i has just ended. In addition, BeGin supports the following final evaluation metrics designed for continual learning:

- Average Performance (AP): Average performance on all tasks after learning all tasks.
- Average Forgetting (AF): Average forgetting on all tasks. We measure the forgetting on task i by the difference between the performance on task i after learning all  tasks and the performance on task i right after learning task i
- Forward Transfer (FWT) : Average forward transfer on tasks. We measure the forward transfer on task i by the difference between the performance on task i after learning task (i-1) and the performance of initialized model on task i.
- Intransigence (INT): Average intransigence on all tasks. We measure the intransigence on task i by the difference between the performances of the Joint model and the the target mode on task i after learning task i. BeGin provides this metric if and only if `full_mode = True`, which simultaneously runs the bare model and the joint model, is enabled.
