"""
This file is used as a regression test with known small networks and properties,
using the command-line interface in the different possible modes.
"""

import subprocess

# Test single instances

print("Test single instance of test_small")
subprocess.run(['python', 'pynever.py', '-o', 'single.csv', 'overapprox', 'single', '-u',
                'examples/data/regtest/test_small.onnx',
                'examples/data/regtest/test_small.vnnlib'])

print("Test single instance of test_ACC")
subprocess.run(['python', 'pynever.py', 'mixed', 'single', '-s',
                'examples/data/regtest/test_ACC.onnx',
                'examples/data/regtest/test_ACC.vnnlib'])

# Test batch instances

print("Test ACC batch instances")
subprocess.run(['python', 'pynever.py', 'complete', 'batch', '-s',
                'examples/data/regtest/instances.csv'])
