Installation and Setup
======================

**pyNeVer** depends on several packages, which are all available via ``pip`` and should be installed automatically.
The packages required for the correct execution are the following:

* *numpy*
* *ortools*
* *onnx*
* *torch*
* *torchvision*
* *pysmt*
* *multipledispatch*

To install **pyNeVer** as an API, run the command:

.. code-block:: bash

    pip install pynever


To run **pyNeVer** as a standalone tool you should clone this repository and create a conda environment

.. code-block:: bash

    git clone https://github.com/nevertools/pyNeVer
    cd pyNeVer

    conda env create -f environment.yml
    conda activate pynever
