"""
This module initializes the configuration variables for the repository
"""
import enum
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open(os.path.join(os.path.dirname(__file__), 'config/configuration.ini'), 'r') as f:
    config_data = dict()
    for line in f:
        l = line.strip('\n').split()
        if '#' not in l and l != []:
            config_data[l[0]] = l[2]

Configuration = enum.Enum('Configuration', config_data)
