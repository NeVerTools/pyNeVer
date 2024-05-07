"""
This file is used as a regression test with known small networks and properties,
using the command-line interface in the different possible modes.
"""

import subprocess

# Test SSBP

print('Test SSBP single')
subprocess.run(['python', 'never2_single.py', '-o', 'ssbp_single.csv',
                'examples/benchmarks/Test/test_small.onnx',
                'examples/benchmarks/Test/test_small.vnnlib',
                '-t', '60', 'ssbp'])

print('Test SSBP batch')
subprocess.run(['python', 'never2_batch.py', '-o', 'ssbp_batch.csv',
                'examples/benchmarks/Test/instances.csv',
                '-t', '60', 'ssbp'])

# Test SSLP

print('Test SSLP single - complete')
subprocess.run(['python', 'never2_single.py', '-o', 'sslp_single1.csv',
                'examples/benchmarks/RL/Networks/cartpole.onnx',
                'examples/benchmarks/RL/Properties/cartpole_case_safe_99.vnnlib',
                'sslp', '-s', 'complete'])

print('Test SSLP single - approx')
subprocess.run(['python', 'never2_single.py', '-o', 'sslp_single2.csv',
                'examples/benchmarks/Test/test_small.onnx',
                'examples/benchmarks/Test/test_small.vnnlib',
                'sslp', '-s', 'overapprox'])

print('Test SSLP batch')
subprocess.run(['python', 'never2_batch.py', '-o', 'sslp_batch.csv',
                'examples/benchmarks/Test/instances.csv',
                'sslp', '-s', 'mixed'])
