"""
This module initializes the configuration variables for the repository
"""
import enum
import os

with open(os.path.join(os.path.dirname(__file__), 'config/configuration.ini'), 'r') as f:
    config_data = {
        l[0]: l[2]
        for l in [line.strip('\n').split() for line in f]
        if l != [] and '#' not in l
    }

Configuration = enum.Enum('Configuration', config_data)
