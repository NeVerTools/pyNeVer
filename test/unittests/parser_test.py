import time

from pynever.strategies.verification.properties import VnnLibProperty

for prop in ['2d_prop.vnnlib', 'prop_3_complex.vnnlib', 'prop_cifar.vnnlib']:
    start = time.perf_counter()
    vnnlib = VnnLibProperty(prop)
    print(f'Parsed property {prop} in {time.perf_counter() - start:4f} seconds')

    print(vnnlib.in_matrix)
    print(vnnlib.in_bias)
    print(vnnlib.out_matrix_list)
    print(vnnlib.out_bias_list)
