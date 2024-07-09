import logging

from pynever.strategies.conversion import representation
from pynever.strategies.conversion.converters.onnx import ONNXConverter
from pynever.strategies.verification.algorithms import SSBPVerification
from pynever.strategies.verification.parameters import SSBPVerificationParameters
from pynever.strategies.verification.properties import VnnLibProperty
from pynever.strategies.verification.ssbp.constants import RefinementStrategy

logger_stream = logging.getLogger("pynever.strategies.verification")
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.INFO)
logger_stream = logging.getLogger("pynever.strategies.bounds_propagation")
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.DEBUG)

property_n = 2
network_n = '2_8'

prop = VnnLibProperty(f'../../examples/benchmarks/ACAS XU/Properties/prop_{property_n}.vnnlib')

onnx_nn = representation.load_network_path(f'../../examples/benchmarks/ACAS XU/Networks/ACAS_XU_{network_n}.onnx')
nn = ONNXConverter().to_neural_network(onnx_nn)

if __name__ == '__main__':
    print(f"Verifying Acas property {property_n} on network {network_n}")
    print(SSBPVerification(SSBPVerificationParameters(heuristic=RefinementStrategy.INPUT_BOUNDS_CHANGE)
                           ).verify(nn, prop))

# 12, 7, 30, 34, 15, 1, 43, 10, 49, 9, 32, 45, 26, 6, 48, 0, 33, 22, 41, 16
# 0, 1, 7, 12, 22, 26, 30, 32, 33, 34, 43, 45