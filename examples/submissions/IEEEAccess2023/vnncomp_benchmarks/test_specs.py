import pynever.networks
import pynever.strategies.smt_reading as smt_reading
import pynever.strategies.conversion as conv
import onnx

spec_path = "vnnlib_specs/"
onnx_path = "onnx_nets/"
spec_ids = ["dubinsrejoin_case_safe_0.vnnlib", "cartpole_case_safe_9.vnnlib", "lunarlander_case_safe_0.vnnlib"]
onnx_ids = ["cartpole.onnx", "dubinsrejoin.onnx", "lunarlander.onnx"]

for i in range(len(spec_ids)):
    vnnlib_parser = smt_reading.SmtPropertyParser(spec_path + spec_ids[i], "X", "Y")
    vnnlib_parser.parse_property()
    print(spec_ids[i])
    print(f"IN_COEF_MAT: {vnnlib_parser.in_coef_mat}")
    print(f"IN_BIAS_MAT: {vnnlib_parser.in_bias_mat}")
    print(f"OUT_COEF_MAT: {vnnlib_parser.out_coef_mat}")
    print(f"OUT_BIAS_MAT: {vnnlib_parser.out_bias_mat}")

for i in range(len(onnx_ids)):
    onnx_net = conv.ONNXNetwork(onnx_ids[i], onnx.load(onnx_path + onnx_ids[i]))
    net = conv.ONNXConverter().to_neural_network(onnx_net)
    assert isinstance(net, pynever.networks.SequentialNetwork)
