Metric
==================


Average Performance (AP)
-----

.. math::

    \mathrm{\mathbf{AP}}=\frac{\sum\limits_{i=1}^{N}\mathrm{\mathbf{M}}_{N,i}}{N}



--------------------


Average Forgetting (AF)
-----

    \mathrm{\mathbf{AF}}=\frac{\sum\limits^{N-1}_{i=1}(\mathrm{\mathbf{M}}_{N,i}-\mathrm{\mathbf{M}}_{i,i})}{N-1}



--------------------


`Intransigence <https://www.naver.com>`_ (INT)
-----

    \mathrm{\mathbf{INT}}=\frac{\sum\limits_{i=1}^{N}(\mathrm{\mathbf{M}}^{\text{Joint}}_{i, i} - \mathrm{\mathbf{M}}_{i, i})}{N}



--------------------

`Forward Transfer <https://www.naver.com>`_ (FWT)
-----

    \mathrm{\mathbf{FWT}}=\frac{\sum\limits^{N}_{i=2}(\mathrm{\mathbf{M}}_{i-1,i}-r_i)}{N-1}


--------------------