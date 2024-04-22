"""
Module never2_batch.py

Provides an entry point for the execution of NeVer2 on a batch of instances

Authors: Stefano Demarchi, Pedro Achete

"""

import os
from argparse import ArgumentParser

import never2_single
from pynever.scripts import cli

parser = ArgumentParser(prog='never2',
                        description='Neural Network verifier',
                        epilog='Università degli Studi di Genova')

# Instances
parser.add_argument('csv', help='Collection of instances to verify')

parser = never2_single.add_options(parser)

if __name__ == '__main__':
    args = vars(parser.parse_args())

    # Clear default log file
    try:
        os.remove('output.csv')
    except OSError:
        pass

    # Check log file specification
    logfile = 'output.csv'
    if 'out' in args.keys():
        logfile = args['out']

    # Execute
    if args['algorithm'] == 'ssbp':
        if not cli.ssbp_verify_batch(args['csv'], logfile, args['timeout'], args['params']):
            exit(1)
    else:
        if not cli.sslp_verify_batch(False, args['csv'], args['strategy'], logfile):
            exit(1)

    exit(0)
