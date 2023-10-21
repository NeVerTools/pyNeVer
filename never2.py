import sys
import platform
import logging

from pynever.scripts import cli

if __name__ == "__main__":

    # # Log to stdout
    logger = logging.getLogger('pynever.strategies.verification')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # CLI
    if len(sys.argv) == 5 and sys.argv[1] == '--batch':
        if not cli.verify_CSV_model(sys.argv[2], sys.argv[3], sys.argv[4]):
            exit(1)
    elif len(sys.argv) == 6 and sys.argv[1] == '--verify':
        writer = open('output.csv', 'w', newline='')
        if not cli.verify_single_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], writer):
            exit(1)
    else:
        cli.show_help()
        exit(1)

    exit(0)
