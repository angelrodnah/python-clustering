python_clustering: all you need for clustering
==============================================

python_clustering is a Python library based on sklearn and numpy. 

It provides a wide range of functionality one might consider using during clustering

Documentation
-------------

Examples
-------------
Some of the examples of using the package are show in the respective 
`folder <https://github.com/IvanReznikov/python-clustering/tree/main/examples>`_

Dependencies
------------

* python = ">=3.8,<3.12"
* pandas = "^1.5.3"
* arff = "^0.9"
* scipy = "^1.10.1"
* requests = "^2.28.2"
* matplotlib = "^3.7.1"
* tqdm = "^4.65.0"
* pyod = "^1.0.7"
* tensorflow = "^2.11.0"

Installation
------------

``pip install python-clustering``

Pytest
-----------
To run tests
We might specify folder or script:

``pytest -q tests``

Development
-----------

python-clustering development takes place on `Github:
<https://github.com/IvanReznikov/python-clustering>`_ 

Please submit bugs
that you encounter to the `issue
tracker <https://github.com/IvanReznikov/python-clustering/issues>`__

Further plans
-------------

1. Implement Task ideas regarding anomaly detection, calculating number
   of cluster using various methods, cluster ensembling and cluster
   similarity
2. Create functionality to detect pathwised cluster
3. Add experimental clustering methods