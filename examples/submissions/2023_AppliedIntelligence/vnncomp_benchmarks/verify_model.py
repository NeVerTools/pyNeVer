import onnx

import pynever.strategies.verification as pyn_ver
import pynever.strategies.conversion as pyn_con
import pynever.strategies.smt_reading as pyn_smt
import argparse
import time
import logging


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", help="Path to the ONNX model to verify.")
    parser.add_argument("-s", "--spec", help="Path to the VNNLIB specification to verify.")
    parser.add_argument("-m", "--mode", help="Mode of operation of the pynever verifier.",
                        choices=["complete", "mixed", "overapprox"])
    parser.add_argument("-l", "--logs", help="Logs filepath.")

    args = parser.parse_args()
    network_path = args.network
    spec_path = args.spec
    ver_mode = args.mode
    logs_filepath = args.logs

    logger_exp_stream = logging.getLogger("pynever.strategies.verification")
    logger_exp_file = logging.getLogger("verification_results")
    exp_file_handler = logging.FileHandler(logs_filepath)
    exp_stream_handler = logging.StreamHandler()
    exp_file_handler.setLevel(logging.INFO)
    exp_stream_handler.setLevel(logging.INFO)
    logger_exp_file.addHandler(exp_file_handler)
    logger_exp_stream.addHandler(exp_stream_handler)
    logger_exp_file.setLevel(logging.INFO)
    logger_exp_stream.setLevel(logging.INFO)

    net_id = network_path.split("/")[-1].replace(".onnx", "")
    spec_id = spec_path.split("/")[-1].replace(".vnnlib", "")
    onnx_network = pyn_con.ONNXNetwork(net_id, onnx.load(network_path))
    pyn_network = pyn_con.ONNXConverter().to_neural_network(onnx_network)

    smt_parser = pyn_smt.SmtPropertyParser(spec_path, "X", "Y")
    smt_parser.parse_property()
    pyn_prop = pyn_ver.NeVerProperty(smt_parser.in_coef_mat, smt_parser.in_bias_mat, smt_parser.out_coef_mat,
                                     smt_parser.out_bias_mat)

    if ver_mode == "complete" or ver_mode == "overapprox":
        verifier = pyn_ver.NeverVerification(ver_mode)
    else:
        verifier = pyn_ver.NeverVerification(ver_mode, [1])

    start = time.perf_counter()
    result = verifier.verify(pyn_network, pyn_prop)
    end = time.perf_counter()

    logger_exp_file.info(f"{net_id},{spec_id},{ver_mode},{result},{end - start}")


if __name__ == "__main__":
    main()
