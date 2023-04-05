import pynever.networks as pyn_net
import pynever.strategies.smt_reading as smt_reading
import pynever.strategies.conversion as pyn_con
import pynever.strategies.verification as pyn_ver
import onnx
import pandas
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime
import torch


def str_to_num_list(in_str: str):
    in_str = in_str.replace(' ', '')
    in_str = in_str.strip('[]')
    in_str = in_str.split(',')
    return [float(f) for f in in_str]


if __name__ == "__main__":
    spec_path = "vnnlib_specs/"
    onnx_path = "onnx_nets/"
    clean_onnx = "clean_onnx/"
    graph_path = "graphs/"
    references_path = "reference_tables/"
    logs_dir = "logs/"

    logger_exp_stream = logging.getLogger("pynever.strategies.verification")
    logger_exp_file = logging.getLogger("test_spec_results")

    log_path = f"{logs_dir}Test_Spec_{datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}.txt"
    exp_file_handler = logging.FileHandler(log_path)
    exp_stream_handler = logging.StreamHandler()

    exp_file_handler.setLevel(logging.INFO)
    exp_stream_handler.setLevel(logging.INFO)

    logger_exp_file.addHandler(exp_file_handler)
    logger_exp_stream.addHandler(exp_stream_handler)

    logger_exp_file.setLevel(logging.INFO)
    logger_exp_stream.setLevel(logging.INFO)

    # cs_ids = ["cartpole", "lunarlander", "dubinsrejoin"]
    cs_ids = ["cartpole"]

    logger_exp_file.info(f"CS,ID,REF_RES,PYN_RES,REF_CE_RESP_SPEC,REF_CE_RESP_TAB,PYN_CES,PYN_OUTS")

    for cs_id in cs_ids:

        details_df = pandas.read_csv(references_path + cs_id + "_details.csv")
        logger_exp_stream.info(details_df.columns)
        sample_dict_list = details_df.to_dict(orient="records")
        total_lb_deviation = []
        total_ub_deviation = []
        is_counterexample = []
        for num_record, record in enumerate(sample_dict_list):

            logger_exp_stream.info(f"Checking {cs_id}_case_{record['RES']}_{num_record}")
            # First we compute the deviation between the input constraints reported in
            # the "truth tables" and the ones in the properties
            corresponding_spec_id = f"{cs_id}_case_{record['RES']}_{num_record}.vnnlib"
            smt_parser = smt_reading.SmtPropertyParser(spec_path + corresponding_spec_id, "X", "Y")
            # print(cs_id, num_record)
            smt_parser.parse_property()

            ref_res = record['RES']

            n_v = 0
            state = str_to_num_list(record['STATE'])
            noise = float(record['NOISE_FRAC'])
            for v in state:
                ref_lb = v - noise
                ref_ub = v + noise
                spec_lb = -smt_parser.in_bias_mat[n_v + 1, 0]
                spec_ub = smt_parser.in_bias_mat[n_v, 0]
                total_lb_deviation.append(abs(abs(ref_lb) - abs(spec_lb)))
                total_ub_deviation.append(abs(abs(ref_ub) - abs(spec_ub)))
                n_v += 2

            # Now we check if the counterexample given in the "truth tables" respect the input constraints as defined
            # both in the tables and the vnnlib specifications.

            if record['COUNTEREXAMPLE'] != 'None':
                counter_example = str_to_num_list(record['COUNTEREXAMPLE'])
                respect_table = True
                respect_spec = True
                n_v = 0
                for i in range(len(counter_example)):
                    v = state[i]
                    c = counter_example[i]
                    ref_lb = v - noise
                    ref_ub = v + noise
                    spec_lb = -smt_parser.in_bias_mat[n_v + 1, 0]
                    spec_ub = smt_parser.in_bias_mat[n_v, 0]
                    # print(f"{spec_lb} < {c} < {spec_ub}")
                    n_v += 2

                    if c < ref_lb or c > ref_ub:
                        respect_table = False

                    if c < spec_lb or c > spec_ub:
                        respect_spec = False
            else:
                respect_spec = None
                respect_table = None

            # Now we check the results of the verification using pynever
            network_path = f"{clean_onnx}{cs_id}.onnx"
            onnx_network = pyn_con.ONNXNetwork(cs_id, onnx.load(network_path))
            pyn_network = pyn_con.ONNXConverter().to_neural_network(onnx_network)
            pyt_network = pyn_con.PyTorchConverter().from_neural_network(pyn_network).pytorch_network

            prop = pyn_ver.NeVerProperty(smt_parser.in_coef_mat, smt_parser.in_bias_mat,
                                         smt_parser.out_coef_mat, smt_parser.out_bias_mat)

            verifier = pyn_ver.NeverVerification("complete", None)
            is_unsafe = verifier.verify(pyn_network, prop)
            pyn_res = 'unsafe' if is_unsafe else 'safe'
            pyn_ces = []
            pyn_ces_outs = []
            if is_unsafe:
                ce_stars = verifier.counterexample_stars
                for star in ce_stars:
                    temp_ce = star.get_samples(1)[0]
                    pyt_temp_ce = torch.from_numpy(temp_ce).squeeze()
                    pyt_temp_ce = pyt_temp_ce.double()
                    pyt_network.double()
                    pyt_out_ce = pyt_network(pyt_temp_ce)
                    pyn_temp_out = pyt_out_ce.detach().numpy()
                    pyn_ces.append(list(temp_ce.squeeze()))
                    pyn_ces_outs.append(list(pyn_temp_out))

            logger_exp_file.info(f'{cs_id},{num_record},{ref_res},{pyn_res},{respect_spec},{respect_table},'
                                 f'"{pyn_ces.__str__()}","{pyn_ces_outs.__str__()}"')

        plt.figure()
        plt.boxplot([total_lb_deviation, total_ub_deviation], labels=["LB", "UB"])
        plt.title("Absolute deviation for LB and UB")
        plt.savefig(graph_path + cs_id + "-abs_lb_ub.eps")
        plt.show()

        is_counterexample = np.array(is_counterexample)
        np.savetxt(f"{references_path}{cs_id}_check_counterexample.csv", is_counterexample, fmt='%s', delimiter=',')







