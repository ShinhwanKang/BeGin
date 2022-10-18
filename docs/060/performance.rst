Performance
==================

Our framework ``BeGin`` provides the evaluator, which computes basic metrics (specifically, accuracy, AUROC, and HITS@K) based on the ground-truth and predicted answers for the queries in Q provided by the loader after each task is processed. 
The basic evaluator can easily be extended by users for additional basic metrics. 


Returns a performance matrix :math:`\text{\mathbf{M}}\in\mathbb^{N\times N}` ...

Accuracy
-----

.. autoclass:: code_evaluator.evaluator.AccuracyEvaluator
    :undoc-members:
    :members:
    :private-members:


--------------------


ROCAUC
-----

.. autoclass:: code_evaluator.evaluator.ROCAUCEvaluator
    :undoc-members:
    :members:
    :private-members:



--------------------


HITS@K
-----

.. autoclass:: code_evaluator.evaluator.HitsEvaluator
    :undoc-members:
    :members:
    :private-members:



--------------------

