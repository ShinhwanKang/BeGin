Performance
==================

Our framework ``BeGin`` provides the evaluator, which computes basic metrics (specifically, accuracy, AUROC, and HITS@K) based on the ground-truth and predicted answers for the queries in Q provided by the loader after each task is processed. 
The basic evaluator can easily be extended by users for additional basic metrics. 


Returns a performance matrix :math:`\mathrm{M}\in\mathbb{R}^{N\times N}`


Examples
--------

.. code-block:: python
    from evaluator import *
    def get_simple_eval_result(self, curr_batch_preds, curr_batch_gts):
            return self.__evaluator.simple_eval(curr_batch_preds, curr_batch_gts)
    
    scenario = DGLNodeClassificationIL(dataset_name=args.dataset_name, num_tasks=args.num_tasks, metric=args.metric, save_path=args.save_path, incr_type='class', task_shuffle=(args.shuffle > 0))
    evaluator_map = {'accuracy': AccuracyEvaluator, 'rocauc': ROCAUCEvaluator, 'hits': HitsEvaluator}
    self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)

    self.eval_fn = lambda x, y: scenario.get_simple_eval_result(x, y)
    

Details are ...

흠.. Scenarios 안에 있어서 ... 특정 ...



Accuracy
--------

.. autoclass:: begin.evaluators.evaluator.AccuracyEvaluator
    :undoc-members:
    :members:
    :private-members:


--------------------


ROCAUC
--------

.. autoclass:: begin.evaluators.evaluator.ROCAUCEvaluator
    :undoc-members:
    :members:
    :private-members:



--------------------


HITS@K
--------

.. autoclass:: begin.evaluators.evaluator.HitsEvaluator
    :undoc-members:
    :members:
    :private-members:



--------------------

