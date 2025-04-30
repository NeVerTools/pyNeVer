"""
Module never2_batch.py

Provides an entry point for the execution of NeVer2 on a batch of instances

Authors: Stefano Demarchi, Pedro Achete

"""

import os
from argparse import ArgumentParser

import never2_launcher
from pynever.scripts import cli

parser = ArgumentParser(prog='NeVer2',
                        description='Neural Network verifier',
                        epilog='Università degli Studi di Genova')

# Instances
parser.add_argument('csv', help='Collection of instances to verify')

parser = never2_launcher.add_options(parser)

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
        try:
            cli.ssbp_verify_batch(args['csv'], '.\\', logfile, args['timeout'], args['params'])
        except NotImplementedError:
            exit(1)
    else:
        try:
            cli.sslp_verify_batch(False, args['csv'], args['strategy'], logfile)
        except NotImplementedError:
            exit(1)

    exit(0)
