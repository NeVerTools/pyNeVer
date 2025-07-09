from pynever.strategies.verification.properties import VnnLibProperty

vnnlib = VnnLibProperty('2d_prop.vnnlib')
print(vnnlib.in_matrix)
print(vnnlib.in_bias)
print(vnnlib.out_matrix_list)
print(vnnlib.out_bias_list)
