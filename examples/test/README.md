This is a test benchmark category. The folder contains  3 .onnx and .vnnlib files used for 
the category. The instance.csv contains the full list of benchmark instances, one per line:


```onnx_file,vnn_lib_file,timeout_secs```
 
The test properties correspond to trivial networks that should ensure the correctness of the
tool and provide a measure of the setup overhead with respect to the verification strategy.

The verification result should be safe (i.e., unsat) for all three benchmarks.