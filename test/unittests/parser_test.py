from pynever.strategies.verification.properties import VnnLibProperty

vnnlib = VnnLibProperty('prop_3_complex.vnnlib')
print(vnnlib.in_matrix)
print(vnnlib.in_bias)
print(vnnlib.out_matrix_list)
print(vnnlib.out_bias_list)
