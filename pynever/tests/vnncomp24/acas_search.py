from pynever.strategies import verification, conversion
from pynever.strategies.conversion import ONNXConverter

prop = verification.NeVerProperty()
prop.from_smt_file('../../../examples/benchmarks/ACAS XU/Properties/prop_3.vnnlib')

onnx_nn = conversion.load_network_path('../../../examples/benchmarks/ACAS XU/Networks/ACAS_XU_1_1.onnx')
nn = ONNXConverter().to_neural_network(onnx_nn)

if __name__ == '__main__':
    print(verification.SearchVerification().verify(nn, prop))
