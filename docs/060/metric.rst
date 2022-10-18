Metric
==================


Average Performance (AP)
--------

.. math:: \mathrm{AP}=\frac{\sum_{i=1}^{N}\mathrm{M}_{N,i}}{N}



--------------------


Average Forgetting (AF)
--------

.. math:: \mathrm{AF}=\frac{\sum_{i=1}^{N-1}\mathrm{M}_{N,i}-\mathrm{M}_{i,i}}{N}


--------------------


`Intransigence <https://www.naver.com>`_ (INT)
--------

.. math:: \mathrm{INT}=\frac{\sum_{i=1}^{N}\mathrm{M}^{Joint}_{i,i}-\mathrm{M}_{i,i}}{N}

    , \text{where}\ \mathrm{M}^{Joint}\ \text{is a basic performance matrix of the Joint model.}


--------------------


`Forward Transfer <https://www.naver.com>`_ (FWT)
--------

.. math:: \mathrm{FWT}=\frac{\sum_{i=2}^{N}\mathrm{M}_{i-1,i}-r_{i}}{N}

    , \text{where}\ r_{i}\ \text{denotes the performance of the initialized model on}\ T_{i}.

--------------------