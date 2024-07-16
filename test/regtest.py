"""
This file is used as a regression test with known small networks and properties,
using the command-line interface in the different possible modes.

"""

import subprocess

# Test SSBP

print('Test SSBP single')
subprocess.run(['python', 'never2_launcher.py', '-o', 'ssbp_single.csv',
                '-t', '60', 'examples/benchmarks/Test/test_small.onnx',
                'examples/benchmarks/Test/test_small.vnnlib', 'ssbp'])

print('Test SSBP batch')
subprocess.run(['python', 'never2_batch.py', '-o', 'ssbp_batch.csv',
                '-t', '60', 'examples/benchmarks/Test/instances.csv',
                'Regression_test_output_ssbp', 'ssbp'])

# Test SSLP

print('Test SSLP single - complete')
subprocess.run(['python', 'never2_launcher.py', '-o', 'sslp_single1.csv',
                '-t', '60', 'examples/benchmarks/RL/Networks/cartpole.onnx',
                'examples/benchmarks/RL/Properties/cartpole_case_safe_99.vnnlib',
                'sslp', '-s', 'complete'])

print('Test SSLP single - approx')
subprocess.run(['python', 'never2_launcher.py', '-o', 'sslp_single2.csv',
                '-t', '60', 'examples/benchmarks/Test/test_small.onnx',
                'examples/benchmarks/Test/test_small.vnnlib', 'sslp',
                '-s', 'overapprox'])

print('Test SSLP batch')
subprocess.run(['python', 'never2_batch.py', '-o', 'sslp_batch.csv', '-t', '60',
                'examples/benchmarks/Test/instances.csv', 'Regression_test_output_sslp',
                'sslp', '-s', 'mixed'])
