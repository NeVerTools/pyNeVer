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

#
#
#
#
# ##### LOGGER INSTANTIATION #####
logger_exp_stream = logging.getLogger("pynever.strategies.verification")
logger_exp_file = logging.getLogger("verification_results")

exp_file_handler = logging.FileHandler(f"{logs_dir}VerResults_{datetime.now().strftime('%m.%d.%Y-%H.%M.%S')}.txt")
exp_stream_handler = logging.StreamHandler()

exp_file_handler.setLevel(logging.INFO)
exp_stream_handler.setLevel(logging.INFO)

logger_exp_file.addHandler(exp_file_handler)
logger_exp_stream.addHandler(exp_stream_handler)

logger_exp_file.setLevel(logging.INFO)
logger_exp_stream.setLevel(logging.INFO)

#
#
#
#
# ##### PARAMETERS DEFINITION #####
onnx_path = "onnx_nets/"
onnx_ids = ["cartpole.onnx", "dubinsrejoin.onnx", "lunarlander.onnx"]
