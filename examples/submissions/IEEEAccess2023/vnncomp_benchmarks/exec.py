import numpy
import onnx
from onnx.reference import ReferenceEvaluator
#
# model_original = onnx.load('onnx_nets/cartpole.onnx')
# onnx.checker.check_model(model_original)
model_clean = onnx.load('clean_onnx/cartpole.onnx')
onnx.checker.check_model(model_clean)

# sess_original = ReferenceEvaluator(model_original)
sess_clean = ReferenceEvaluator(model_clean)

counterexample = numpy.array([-0.25266144, -0.24881521, -0.04145561, -0.0261547])
feeds_cnt = {'X': counterexample}

# print(sess_original.run(None, feeds))
print(sess_clean.run(None, feeds_cnt))
