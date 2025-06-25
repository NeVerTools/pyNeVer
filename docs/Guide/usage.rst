Command-line interface
======================

| To verify `VNN-LIB <https://www.vnnlib.org>`_ specifications on ONNX models we provide two scripts:
| one for single instances and another one for multiple instances.

To verify a single instance run

.. code-block:: bash

    python never2_launcher.py [-o OUTPUT] [-t TIMEOUT] model.onnx property.vnnlib {sslp|ssbp}

For multiple instances collected in a CSV file run

.. code-block:: bash

    python never2_batch.py [-o OUTPUT] [-t TIMEOUT] instances.csv {sslp|ssbp}

* The ``-o`` option should be used to specify the output CSV file to save results, otherwise it will be generated in the same directory
* The ``-t`` option specifies the timeout for each run
* ``sslp`` and ``ssbp`` are the two algorithms employed by **NeVer2**:

  * SSLP (Star-set with Linear Programming) is our first algorithm based on star sets presented in `this paper <https://link.springer.com/article/10.1007/s00500-024-09907-5>`_.
  * SSBP (Star-set with Bounds Propagation) enhances SSLP with an abstraction-refinement search and symbolic interval propagation. This is the algorithm used in VNNCOMP 2024.

Supported layers
----------------------

At present the **pyNeVer** package supports abstraction and verification of fully connected and convolutional
neural networks with ReLU activation functions.

Training and conversion support all the layers supported by the `VNN-LIB standard <https://easychair.org/publications/paper/Qgdn>`_.

The :ref:`conversion <conversion-ref>` package provides the capabilities for the conversion of PyTorch and ONNX
networks: therefore this kind of networks can be loaded using the respective frameworks and then converted to the
internal representation used by **pyNeVer**.

The properties for the verification and abstraction of the networks must be defined either in python code following
the specification which can be found in the documentation, or via an SMT-LIB file compliant to the
`VNN-LIB <https://www.vnnlib.org>`_ standard.

API
============

In the `notebooks <https://github.com/NeVerTools/pyNeVer/tree/main/examples/notebooks>`_ directory there are four Jupyter Notebooks that illustrate how to use **pyNeVer** as an API to design, train and verify neural networks.

- The `first notebook <https://github.com/NeVerTools/pyNeVer/blob/main/examples/notebooks/00%20-%20Networks.ipynb>`_ covers the classes and methods to build networks
- The `second notebook <https://github.com/NeVerTools/pyNeVer/blob/main/examples/notebooks/01%20-%20Training.ipynb>`_ covers the learning strategy to train and test a network
- The `third notebook <https://github.com/NeVerTools/pyNeVer/blob/main/examples/notebooks/02%20-%20Safety%20specifications.ipynb>`_ explains how to build a safety specification to define a verification problem
- The `fourth notebook <https://github.com/NeVerTools/pyNeVer/blob/main/examples/notebooks/03%20-%20Verification.ipynb>`_ explains our verification algorithms and covers how to instantiate and execute verification
