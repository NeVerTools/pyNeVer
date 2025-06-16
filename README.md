# pyNeVer

[![PyPI - Version](https://img.shields.io/pypi/v/pynever.svg)](https://pypi.org/project/pynever)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pynever.svg)](https://pypi.org/project/pynever)

-----

Neural networks Verifier (__NeVer 2__) is a tool for the design, training and verification of neural networks.
It supports feed-forward and residual neural networks with ReLU and activation functions.
__pyNeVer__ is the corresponding python package providing all the main capabilities of __NeVer 2__
and can be easily installed using pip. 

Installation and setup
----------------------

__pyNeVer__ depends on several packages, which are all available via pip and should be installed automatically. 
The packages required for the correct execution are the following:

* _numpy_
* _ortools_
* _onnx_
* _torch_
* _torchvision_
* _pysmt_
* _multipledispatch_

To install __pyNeVer__ as an API, run the command:
```bash
pip install pynever
```

To run __pyNeVer__ as a standalone tool you should clone this repository and create a conda environment
```bash
git clone https://github.com/nevertools/pyNeVer
cd pyNeVer

conda env create -f environment.yml
conda activate pynever
```

Command-line interface
----------------------
To verify [VNN-LIB](https://www.vnnlib.org) specifications on ONNX models we provide two scripts: one for single instances and another one for multiple instances.
To verify a single instance run
```bash
python never2_launcher.py [-o OUTPUT] [-t TIMEOUT] model.onnx property.vnnlib {sslp|ssbp}
```

For multiple instances collected in a CSV file run
```bash
python never2_batch.py [-o OUTPUT] [-t TIMEOUT] instances.csv {sslp|ssbp}
```
* The -o option should be used to specify the output CSV file to save results, otherwise it will be generated in the same directory
* The -t option specifies the timeout for each run
* sslp and ssbp are the two algorithms employed by _NeVer2_:
  * SSLP (Star-set with Linear Programming) is our first algorithm based on star sets presented in [this paper](https://link.springer.com/article/10.1007/s00500-024-09907-5).
  * SSBP (Star-set with Bounds Propagation) enhances SSLP with an abstraction-refinement search and symbolic interval propagation. This is the algorithm used in VNNCOMP 2024.

API
---------------------
In the [notebooks](examples/notebooks) directory there are four Jupyter Notebooks that illustrate how to use _pyNever_ as an API to design, train and verify neural networks.

- The [first notebook](examples/notebooks/00%20-%20Networks.ipynb) covers the classes and methods to build networks
- The [second notebook](examples/notebooks/01%20-%20Training.ipynb) covers the learning strategy to train and test a network
- The [third notebook](examples/notebooks/02%20-%20Safety%20specifications.ipynb) explains how to build a safety specification to define a verification problem
- The [fourth notebook](examples/notebooks/03%20-%20Verification.ipynb) explains our verification algorithms and covers how to instantiate and execute verification

Supported layers
----------------------

At present the __pyNeVer__ package supports abstraction and verification of fully connected and convolutional 
neural networks with ReLU activation functions.

Training and conversion support all the layers supported by [VNN-LIB](https://easychair.org/publications/paper/Qgdn).

The [conversion](pynever/strategies/conversion) package provides the capabilities for the conversion of PyTorch and ONNX
networks: therefore this kind of networks can be loaded using the respective frameworks and then converted to the
internal representation used by __pyNeVer__.  

The properties for the verification and abstraction of the networks must be defined either in python code following
the specification which can be found in the documentation, or via an SMT-LIB file compliant to the 
[VNN-LIB](https://www.vnnlib.org) standard.

Contributors
----------------------

The main contributors of pyNeVer are __Dario Guidotti__ and __Stefano Demarchi__, under the supervision of Professors
__Armando Tacchella__ and __Luca Pulina__.  
A significant contribution for the participation in VNN-COMP 2024 was
the help of __Elena Botoeva__.

_Other contributors_:

* __Andrea Gimelli__ - Bound propagation integration
* __Pedro Henrique Simão Achete__ - Command-line interface and convolutional linearization
* __Karim Pedemonte__ - Design and refactoring

To contribute to this project, start by looking at the [CONTRIBUTING](CONTRIBUTING.md) file!

Publications
----------------------

If you use __NeVer2__ or __pyNeVer__ in your work, **please kindly cite our papers**. Here you can find 
the list of BibTeX entries.

```
@article{demarchi2024never2,
	author = {Demarchi, S. and Guidotti, D. and Pulina, L. and Tacchella, A.},
	journal = {Soft Computing},
	number = {19},
	pages = {11647-11665},
	title = {{NeVer2}: learning and verification of neural networks},
	volume = {28},
	year = {2024}
}

@inproceedings{demarchi2024improving,
	author = {Demarchi, S. and Gimelli, A. and Tacchella, A.},
	booktitle = {{ECMS} International Conference on Modelling and Simulation},
	title = {Improving Abstract Propagation for Verification of Neural Networks},
	year = {2024}
}

@inproceedings{demarchi2022formal,
	author = {Demarchi, S. and Guidotti, D. and Pitto, A. and Tacchella, A.},
	booktitle = {{ECMS} International Conference on Modelling and Simulation},
	title = {Formal Verification of Neural Networks: {A} Case Study About Adaptive Cruise Control},
	year = {2022}
}

@inproceedings{guidotti2021pynever,
    author={Guidotti, D. and Pulina, L. and Tacchella, A.},
    booktitle={International Symposium on Automated Technology for Verification and Analysis},
    title={pynever: A framework for learning and verification of neural networks},
    year={2021},
}
```