Metrics for CL
==================

For the final evaluation metric, BeGin provides the following metrics.
In the mathematical expression, :math:`\mathrm{M}_{i,j}` indicates the performance on on j-th task after the i-th task is processed,
:math:`\mathrm{M}^{joint}` is a basic performance matrix of the joint model,
and :math:`r_i` denotes the performance of a randomly initialized model on i-th task.

Average Performance (AP)
--------------------------------

Average performance on each task after learning all tasks.

.. math:: \mathrm{AP}=\frac{\sum_{i=1}^{N}\mathrm{M}_{N,i}}{N}


--------------------


Average Forgetting (AF)
--------------------------------

Average forgetting on each task after learning all tasks.
We measure the forgetting on the i-th task by the difference between the performance on the i-th task after learning all tasks
and the performance on the i-th task right after learning the i-th task.

.. math:: \mathrm{AF}=\frac{\sum_{i=1}^{N-1}\mathrm{M}_{N,i}-\mathrm{M}_{i,i}}{N}


--------------------


`Intransigence <https://openaccess.thecvf.com/content_ECCV_2018/papers/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.pdf>`_ (INT)
--------------------------------------------------------------------------------------------------------------------------------------

Average intransigence on each task. We measure the intransigence on the i-th task
by the difference between the performances of the Joint model and the target model on the i-th task after learning the i-th task.

.. math:: \mathrm{INT}=\frac{\sum_{i=1}^{N}\mathrm{M}^{Joint}_{i,i}-\mathrm{M}_{i,i}}{N}

--------------------


`Forward Transfer <https://arxiv.org/pdf/1706.08840.pdf>`_ (FWT)
-------------------------------------------------------------------

Average forward transfer on each task.
We measure the forward transfer on the i-th task by the difference between the performance on the i-th task
after learning (i−1)-th task and the performance :math:`r_i` on the i-th task without any learning.

.. math:: \mathrm{FWT}=\frac{\sum_{i=2}^{N}\mathrm{M}_{i-1,i}-r_{i}}{N}

