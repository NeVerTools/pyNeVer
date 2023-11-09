import argparse
import os

from pynever.scripts import cli

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('strategy', choices=['approx', 'mixed', 'complete'],
                        default='complete', help='Verification strategy to use, uses complete if not provided')
    parser.add_argument('-o', '--out', default='output.csv',
                        help='Output file for execution log')

    subparsers = parser.add_subparsers(title='Execution mode',
                                       description='verify a single property or a batch of instances')

    single_parser = subparsers.add_parser('single', help='Verify a single instance')
    group = single_parser.add_argument_group('Property type', 'expected structure of the property')
    prop_type = group.add_mutually_exclusive_group(required=True)
    prop_type.add_argument('-s', action='store_true', help='Safety property')
    prop_type.add_argument('-u', action='store_true', help='Unsafety property')
    single_parser.add_argument('model', help='Network model in ONNX format')
    single_parser.add_argument('property', help='Property specification in VNN-LIB format')

    batch_parser = subparsers.add_parser('batch', help='Verify a batch of instances')
    group = batch_parser.add_argument_group('Property type', 'expected structure of the property')
    prop_type = group.add_mutually_exclusive_group(required=True)
    prop_type.add_argument('-s', action='store_true', help='Safety property')
    prop_type.add_argument('-u', action='store_true', help='Unsafety property')
    batch_parser.add_argument('csv', help='CSV file that specifies the instances')

    args = vars(parser.parse_args())
    if args['s']:
        safety = True
    else:
        safety = False

    # Clear default log file
    try:
        os.remove('output.csv')
    except OSError:
        pass

    # Check log file specification
    logfile = 'output.csv'
    if 'out' in args.keys():
        logfile = args['out']

    # Execution
    if 'model' in args.keys():
        if not cli.verify_single_model(safety, args['model'], args['property'], args['strategy'], logfile):
            exit(1)

    elif 'csv' in args.keys():
        if not cli.verify_CSV_batch(safety, args['csv'], args['strategy'], logfile):
            exit(1)

    else:
        parser.print_help()
        exit(1)

    exit(0)
