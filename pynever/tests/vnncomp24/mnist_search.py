import logging

from pynever.strategies import verification, conversion
from pynever.strategies.conversion import ONNXConverter

logger_stream = logging.getLogger("pynever.strategies.verification")
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.INFO)
logger_stream = logging.getLogger("pynever.strategies.bp.bounds_manager")
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.INFO)

prop = verification.NeVerProperty()
prop.from_smt_file('../../../examples/benchmarks/mnist_fc/Properties/prop_1_0.03.vnnlib')

onnx_nn = conversion.load_network_path('../../../examples/benchmarks/mnist_fc/Networks/mnist-net_256x4.onnx')
nn = ONNXConverter().to_neural_network(onnx_nn)

if __name__ == '__main__':
    print(verification.SearchVerification().verify(nn, prop))