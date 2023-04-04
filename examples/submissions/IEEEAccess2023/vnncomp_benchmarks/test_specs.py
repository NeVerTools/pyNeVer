import pynever.networks
import pynever.strategies.smt_reading as smt_reading
import pynever.strategies.conversion as conv
import onnx
import pandas
import matplotlib.pyplot as plt
import numpy as np


def str_to_num_list(in_str: str):
    in_str = in_str.replace(' ', '')
    in_str = in_str.strip('[]')
    in_str = in_str.split(',')
    return [float(f) for f in in_str]


spec_path = "vnnlib_specs/"
onnx_path = "onnx_nets/"
clean_onnx = "clean_onnx/"
graph_path = "graphs/"
references_path = "reference_tables/"
cs_ids = ["cartpole", "dubinsrejoin", "lunarlander"]

for cs_id in cs_ids:

    details_df = pandas.read_csv(references_path + cs_id + "_details.csv")
    print(details_df.columns)
    sample_dict_list = details_df.to_dict(orient="records")
    total_lb_deviation = []
    total_ub_deviation = []
    is_counterexample = []
    for num_record, record in enumerate(sample_dict_list):

        # First we compute the deviation between the input constraints reported in
        # the "truth tables" and the ones in the properties
        corresponding_spec_id = f"{cs_id}_case_{record['RES']}_{num_record}.vnnlib"
        smt_parser = smt_reading.SmtPropertyParser(spec_path + corresponding_spec_id, "X", "Y")
        print(cs_id, num_record)
        smt_parser.parse_property()

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

        # Now we check if the counterexample given in the "truth tables" respect the input constraints as defined both
        # in the tables and the vnnlib specifications.

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

            is_counterexample.append([respect_table, respect_spec])
        else:
            is_counterexample.append([None, None])

    plt.boxplot([total_lb_deviation, total_ub_deviation], labels=["LB", "UB"])
    plt.title("Absolute deviation for LB and UB")
    plt.savefig(graph_path + cs_id + "-abs_lb_ub.eps")
    plt.show()

    is_counterexample = np.array(is_counterexample)
    np.savetxt(f"{references_path}{cs_id}_check_counterexample.csv", is_counterexample, fmt='%s', delimiter=',')







