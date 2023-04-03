import pynever.strategies.smt_reading as smt_reading


spec_path = "vnnlib_specs/dubinsrejoin_case_safe_0.vnnlib"
# spec_path = "vnnlib_specs/cartpole_case_safe_9.vnnlib"

vnnlib_parser = smt_reading.SmtPropertyParser(spec_path, "X", "Y")
vnnlib_parser.parse_property()
print(vnnlib_parser.in_coef_mat, vnnlib_parser.in_bias_mat)
print(vnnlib_parser.out_coef_mat, vnnlib_parser.out_bias_mat)