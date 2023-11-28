import pynever.networks as pyn_net
import pynever.strategies.conversion as pyn_conv
import pynever.strategies.verification as pyn_ver
import pynever.nodes as pyn_nodes
import pynever.strategies.smt_reading as pyn_smt
import onnx
import logging
import os
import copy
from datetime import datetime


#
#
#
#
# ##### DIRECTORIES CREATION #####
logs_dir = "logs/"
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)

clean_dir = "clean_onnx/"
if not os.path.exists(clean_dir):
    os.mkdir(clean_dir)

#
#
#
#
# ##### PARAMETERS DEFINITION #####
onnx_path = "onnx_nets/"
onnx_ids = ["cartpole.onnx", "dubinsrejoin.onnx", "lunarlander.onnx"]

#
#
#
#
# ##### MODELS CLEANING #####

for onnx_id in onnx_ids:
    net_id = onnx_id.replace(".onnx", "")
    onnx_model = pyn_conv.ONNXNetwork(net_id, onnx.load(onnx_path + onnx_id))
    temp_net = pyn_conv.ONNXConverter().to_neural_network(onnx_model)

    # Network cleaning: we assume only the FC nodes and the ReLU nodes are relevant
    assert isinstance(temp_net, pyn_net.SequentialNetwork)
    current_node = temp_net.get_first_node()
    clean_net = pyn_net.SequentialNetwork(net_id, "X")
    node_counter = 0
    while current_node is not None:

        if isinstance(current_node, pyn_nodes.FullyConnectedNode):
            clean_node = copy.deepcopy(current_node)
            clean_node.identifier = f"FC_{node_counter}"
            clean_net.add_node(clean_node)
            node_counter += 1
        elif isinstance(current_node, pyn_nodes.ReLUNode):
            clean_node = copy.deepcopy(current_node)
            clean_node.identifier = f"ReLU_{node_counter}"
            clean_net.add_node(clean_node)
            node_counter += 1

        else:
            pass

        current_node = temp_net.get_next_node(current_node)

    onnx.save(pyn_conv.ONNXConverter().from_neural_network(clean_net).onnx_network, clean_dir + clean_net.identifier +
              ".onnx")


