"""
This module reads a VNNLIB file containing a robustness specification
"""


def read_vnnlib(filename: str):
    lbs = []
    ubs = []
    label = int(filename.split('_')[-1].replace('.vnnlib', ''))

    with open(filename, 'r') as vnnlib_file:
        # Input condition
        for line in vnnlib_file:
            if '>=' in line:
                lbs.append(float(line.split()[-1].replace('))', '')))
            elif '<=' in line:
                ubs.append(float(line.split()[-1].replace('))', '')))
            elif 'or' in line:
                # We reached the output condition
                break

    return lbs, ubs, label
