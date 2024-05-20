from pynever.strategies import verification, conversion
from pynever.strategies.conversion import ONNXConverter

prop = verification.NeVerProperty()
prop.from_smt_file('../../../examples/Benchmarks/ACAS XU/Properties/prop_3.vnnlib')

onnx_nn = conversion.load_network_path('../../../examples/Benchmarks/ACAS XU/Networks/ACAS_XU_1_1.onnx')
nn = ONNXConverter().to_neural_network(onnx_nn)

print(verification.SearchVerification().verify(nn, prop))
