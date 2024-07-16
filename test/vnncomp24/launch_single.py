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

benchmark = 'ACAS XU'
network_n = 'ACAS_XU_2_8'
property_n = 'prop_2'

prop = VnnLibProperty(f'../../examples/benchmarks/{benchmark}/Properties/{property_n}.vnnlib')

onnx_nn = representation.load_network_path(f'../../examples/benchmarks/{benchmark}/Networks/{network_n}.onnx')
nn = ONNXConverter().to_neural_network(onnx_nn)

if __name__ == '__main__':
    print(f"Verifying {benchmark} property {property_n} on network {network_n}")
    print(SSBPVerification(SSBPVerificationParameters(heuristic=RefinementStrategy.INPUT_BOUNDS_CHANGE)
                           ).verify(nn, prop))
