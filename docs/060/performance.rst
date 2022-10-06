Basic Performance Metrics
==========================

Our framework ``BeGin`` provides the evaluator, which computes basic metrics (specifically, accuracy, AUROC, and HITS@K) based on the ground-truth and predicted answers for the queries in Q provided by the loader after each task is processed. 
The basic evaluator can easily be extended by users for additional basic metrics. 

BaseEvaluator
----------------

.. autoclass:: begin.evaluators.BaseEvaluator
    :undoc-members:
    :members:
    :private-members:


--------------------

Accuracy
----------------

.. autoclass:: begin.evaluators.AccuracyEvaluator
    :undoc-members:
    :members:
    :private-members:


--------------------

ROCAUC
----------------

.. autoclass:: begin.evaluators.ROCAUCEvaluator
    :undoc-members:
    :members:
    :private-members:



--------------------


HITS@K
----------------

.. autoclass:: begin.evaluators.HitsEvaluator
    :undoc-members:
    :members:
    :private-members:


