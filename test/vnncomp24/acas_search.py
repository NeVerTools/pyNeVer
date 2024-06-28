import logging

from pynever.strategies.conversion import representation
from pynever.strategies.conversion.converters.onnx import ONNXConverter
from pynever.strategies.verification.algorithms import SSBPVerification
from pynever.strategies.verification.parameters import SSBPVerificationParameters
from pynever.strategies.verification.properties import VnnLibProperty

logger_stream = logging.getLogger("pynever.strategies.verification")
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.INFO)
logger_stream = logging.getLogger("pynever.strategies.bounds_propagation")
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.DEBUG)

prop = VnnLibProperty('../../examples/benchmarks/ACAS XU/Properties/prop_2.vnnlib')

onnx_nn = representation.load_network_path('../../examples/benchmarks/ACAS XU/Networks/ACAS_XU_2_2.onnx')
nn = ONNXConverter().to_neural_network(onnx_nn)

if __name__ == '__main__':
    print(SSBPVerification(SSBPVerificationParameters()).verify(nn, prop))

# 12, 7, 30, 34, 15, 1, 43, 10, 49, 9, 32, 45, 26, 6, 48, 0, 33, 22, 41, 16
# 0, 1, 7, 12, 22, 26, 30, 32, 33, 34, 43, 45