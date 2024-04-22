"""
This file is used as a regression test with known small networks and properties,
using the command-line interface in the different possible modes.
"""

import subprocess

# Test SSBP

print('Test SSBP')
subprocess.run(['python', 'never2_single.py', '-o', 'ssbp.csv',
                'examples/benchmarks/Test/test_small.onnx',
                'examples/benchmarks/Test/test_small.vnnlib',
                '-t', '60', 'ssbp'])

# Test single instances

print("Test single instance of test_small")
subprocess.run(['python', 'pynever.py', '-o', 'single.csv', 'overapprox', 'single', '-u',
                'examples/benchmarks/Test/test_small.onnx',
                'examples/benchmarks/Test/test_small.vnnlib'])

print("Test single instance of test_ACC")
subprocess.run(['python', 'pynever.py', 'mixed', 'single', '-u',
                'examples/benchmarks/ACC/Networks/NET_0_1.5_5.onnx',
                'examples/benchmarks/ACC/Properties/prop_far0_eps0.vnnlib'])

# Test batch instances

print("Test ACC batch instances")
subprocess.run(['python', 'pynever.py', 'complete', 'batch', '-u',
                'examples/benchmarks/ACC/instances.csv'])
