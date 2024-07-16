# pyNeVer

[![PyPI - Version](https://img.shields.io/pypi/v/pynever.svg)](https://pypi.org/project/pynever)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pynever.svg)](https://pypi.org/project/pynever)

-----

Neural networks Verifier (__NeVer 2__) is a tool for the design, training and verification of neural networks.
It supports sequential fully connected and convolutional neural networks with ReLU and Sigmoid activation functions.
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

To install __pyNeVer__, just run the command:

```bash
pip install pynever
```

To run some examples, further packages may be required. If an example requires a specific package, it will be 
detailed in the example directory.

[//]: # (#### DOCUMENTATION)

[//]: # (The documentation related to the __pyNeVer__ package can be found in the directory docs/pynever/ as html files.)

Supported inputs
----------------------

At present the __pyNeVer__ package supports only the abstraction and verification of fully connected and convolutional 
neural networks with ReLU and Sigmoid activation functions. The training and conversion supports also batch normalization
layers. A network with batchnorm layers following fully connected layers can be converted to a "pure" fully connected
neural networks using the capabilities provided in the [utilities.py](pynever/utilities.py) module.  
The [conversion](pynever/strategies/conversion) package provides the capabilities for the conversion of PyTorch and ONNX
networks: therefore this kind of networks can be loaded using the respective frameworks and then converted to the
internal representation used by __pyNeVer__.  
The properties for the verification and abstraction of the networks must be defined either in python code following
the specification which can be found in the documentation, or via an SMT-LIB file compliant to the 
[VNN-LIB](http://www.vnnlib.org) standard.

Examples
----------------------

**NB: All the scripts should be executed INSIDE the related directory!**

***All the examples described below are guaranteed to work until [Release v0.1.1a4](https://github.com/NeVerTools/pyNeVer/releases/tag/v0.1.1a4). 
After this release, changes in the interface structure may add inconsistencies between test scripts and API, so
the old examples will be removed and new examples will be created in future releases.***

* The directory examples/ contains some examples of application of the __pyNeVer__ package. In particular the 
[jupyter notebook](examples/notebooks/bidimensional_example_with_sigmoid.ipynb) shows a graphical example of the 
application of the abstraction module for the reachability of a small network with bi-dimensional input and outputs.  
  
* The [pruning_example.py](examples/pruning_example/pruning_example.py) script show how to train and prune some small
fully connected neural networks with relu activation function. It also show how it is possible to combine batch norm
layer and fully connected layers to make the networks compliant with the requirements of the verification and 
abstraction modules.  

* The directory examples/submissions/ATVA2021 contains the experimental setup used for the experimental evaluation
in our ATVA2021 paper. The experiments can be easily replicated by executing the python scripts 
[acas_experiment.py](examples/submissions/2021_ATVA/acas_experiments.py) from within the ATVA2021/ directory. 
The log files will be generated and will be saved in the logs/ directory.

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

Publications
----------------------

If you use __NeVer2__ or __pyNeVer__ in your work, **please kindly cite our papers**. Here you can find 
the list of BibTeX entries.

```
@article{demarchi2024never2,
  title={NeVer2: Learning and Verification of Neural Networks},
  author={Demarchi, Stefano and Guidotti, Dario and Pulina, Luca and Tacchella, Armando},
  journal={Soft Computing},
  year={2024}
}

@inproceedings{demarchi2022formal,
  title={Formal Verification Of Neural Networks: A Case Study About Adaptive Cruise Control.},
  author={Demarchi, Stefano and Guidotti, Dario and Pitto, Andrea and Tacchella, Armando},
  booktitle={ECMS},
  pages={310--316},
  year={2022}
}

@inproceedings{guidotti2021pynever,
  title={pynever: A framework for learning and verification of neural networks},
  author={Guidotti, Dario and Pulina, Luca and Tacchella, Armando},
  booktitle={Automated Technology for Verification and Analysis: 19th International Symposium, ATVA 2021, Gold Coast, QLD, Australia, October 18--22, 2021, Proceedings 19},
  pages={357--363},
  year={2021},
  organization={Springer}
}

```