import csv

from pynever.scripts import cli

if __name__ == "__main__":
    if not cli.verify_CSV_model('-u', 'ACC/instances.csv', 'complete'):
        exit(1)

    reader = csv.reader(open("output.csv", 'r', newline=''))

    for row in reader:
        if row[3] != "Falsified":
            exit(1)

    exit(0)
