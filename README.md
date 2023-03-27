# pyNeVer

Neural networks Verifier (__NeVer 2__) is a tool for the training, pruning and verification of neural networks.
At present it supports sequential fully connected neural networks with ReLU and Sigmoid activation functions.
__pyNeVer__ is the corresponding python package providing all the main capabilities of the __NeVer 2__ tool
and can be easily installed using pip. The PyPI project page can be found at <https://pypi.org/project/pyNeVer/>
whereas the github repository can be found at <https://github.com/NeVerTools/pyNeVer>.

#### REQUIREMENTS AND INSTALLATION
__pyNeVer__ depends on several packages, which should be installed automatically. The packages required for the
correct execution are the following:

* _numpy_
* _ortools_
* _onnx_
* _torch_
* _torchvision_
* _pysmt_

All the above packages are available via pip.

To install __pyNeVer__, run the command:

```bash
pip install pynever
```

To run some examples, further packages may be required. If an example requires a specific package, it will be 
detailed in the example directory.

#### DOCUMENTATION
The documentation related to the __pyNeVer__ package can be found in the directory docs/pynever/ as html files.

#### SUPPORTED INPUTS
At present the __pyNeVer__ package supports only the abstraction and verification of fully connected neural networks 
with ReLU and Sigmoid activation functions. The training, pruning and conversion supports also batch normalization
layers. A network with batchnorm layers following fully connected layers can be converted to a "pure" fully connected
neural networks using the capabilities provided in the [utilities.py](pynever/utilities.py) module.  
The [conversion.py](pynever/strategies/conversion.py) provides the capabilities for the conversion of PyTorch and ONNX
networks: therefore this kind of networks can be loaded using the respective frameworks and then converted to the
internal representation used by __pyNeVer__.
The properties for the verification and abstraction of the networks must be defined either in python code following
the specification which can be found in the documentation, or via an SMT-LIB file compliant to the 
[VNN-LIB](http://vnnlib.org) standard. Examples of the python specification of the properties can be found in all the 
scripts in the directory examples/submissions/ATVA2021/.

#### EXAMPLES
**NB: All the scripts should be executed INSIDE the related directory!**  

* The directory examples/ contains some examples of application of the __pyNeVer__ package. In particular the 
[jupyter notebook](examples/notebooks/bidimensional_example_with_sigmoid.ipynb) shows a graphical example of the 
application of the abstraction module for the reachability of a small network with bi-dimensional input and outputs.  
  
* The [pruning_example.py](examples/pruning_example/pruning_example.py) script show how to train and prune some small
fully connected neural networks with relu activation function. It also show how it is possible to combine batch norm
layer and fully connected layers to make the networks compliant with the requirements of the verification and 
abstraction modules.  

* The directory examples/submissions/ATVA2021 contains the experimental setup used for the experimental evaluation
in our ATVA2021 paper. The experiments can be easily replicated by executing the python scripts 
[acas_experiment.py](examples/submissions/ATVA2021/acas_experiments.py) from within the ATVA2021/ directory. 
The log files will be generated and will be saved in the logs/ directory.

#### CONTRIBUTORS
The main contributors of pyNeVer are __Dario Guidotti__ and __Armando Tacchella__, further contributions are provided 
by __Stefano Demarchi__.

_Students contributions_:

* __Alessandro Drago__ - TensorFlow conversion