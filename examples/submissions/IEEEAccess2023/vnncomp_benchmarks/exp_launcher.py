import time

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

log_path = f"{logs_dir}VerResults_{datetime.now().strftime('%m.%d.%Y-%H.%M.%S')}.txt"
exp_file_handler = logging.FileHandler(log_path)
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
onnx_path = "clean_onnx/"
onnx_ids = ["cartpole.onnx"]  # , "dubinsrejoin.onnx", "lunarlander.onnx"]
verification_modes = ["complete"]
specs_path = "vnnlib_specs/"
specs_ids = os.listdir(specs_path)
timeout = 60

logger_exp_file.info("NET_ID,SPEC_ID,VER_MODE,RESULT,TIME")

for onnx_id in onnx_ids:
    cs_id = onnx_id.replace(".onnx", "")
    corresponding_ids = [temp_id for temp_id in specs_ids if cs_id in temp_id]
    for spec_id in corresponding_ids:

        clean_spec_id = spec_id.replace(".vnnlib", "")
        for verification_mode in verification_modes:
            logger_exp_stream.info(f"CHECKING Property: {clean_spec_id} ON Network: {cs_id} "
                                   f"WITH Mode: {verification_mode}")
            command = f"timeout {timeout}s python verify_model.py -n {onnx_path + onnx_id} -s {specs_path + spec_id} " \
                      f"-m {verification_mode} -l {log_path}"

            start = time.perf_counter()
            res = os.system(command)
            end = time.perf_counter()
            if res != 0:
                logger_exp_file.info(f"{cs_id},{clean_spec_id},{verification_mode},-,-")

