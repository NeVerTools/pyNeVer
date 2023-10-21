from pynever.scripts import cli
import csv

if __name__ == "__main__":

    if not cli.verify_CSV_model("-u", "instances.csv", "complete"):
        exit(1)

    reader = csv.reader(open("output.csv", 'r', newline=''))

    for row in reader:
        if row[3] != "Falsified":
            exit(1)

    exit(0)
