"""
Module never2_launcher.py

Provides an entry point for the execution of NeVer2 on a single instance

Authors: Stefano Demarchi, Pedro Achete

"""

import os
from argparse import ArgumentParser

from pynever.scripts import cli


def add_options(p: ArgumentParser):
    """
    Common options for the execution of NeVer2

    """

    # Options
    p.add_argument('-o', '--out', type=str,
                   default='output.csv', help='output file for execution log')
    p.add_argument('-t', '--timeout', type=int, default=300,
                   help='execution timeout in seconds')

    # Algorithm
    algorithm = p.add_subparsers(dest='algorithm', description='Verification algorithm to use')

    # SSBP
    ssbp = algorithm.add_parser('ssbp', description='Starset with bounds propagation')
    ssbp.add_argument('-p', '--params', nargs='?', default='', metavar='FILE',
                      help='JSON file with parameters')

    # SSLP
    sslp = algorithm.add_parser('sslp', description='Starset with linear programs')
    sslp.add_argument('-s', '--strategy', choices=['overapprox', 'mixed', 'complete'], metavar='STRATEGY',
                      default='complete', help='Verification strategy to use, complete by default')

    return p


if __name__ == '__main__':
    parser = ArgumentParser(prog='NeVer2',
                            description='Neural Network verifier',
                            epilog='Università degli Studi di Genova')

    # Instance
    parser.add_argument('model', help='network model in ONNX format')
    parser.add_argument('property', help='property specification in VNN-LIB format')

    parser = add_options(parser)

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
            cli.ssbp_verify_single(args['model'], args['property'], './', logfile, args['timeout'], args['params'])
        except NotImplementedError:
            exit(1)
    else:
        try:
            cli.sslp_verify_single(False, args['model'], args['property'], args['strategy'], logfile)
        except NotImplementedError:
            exit(1)

    exit(0)
