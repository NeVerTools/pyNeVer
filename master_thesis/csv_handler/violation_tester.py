import pandas as pd
import logging
violation_logger = logging.getLogger("pynever/master_thesis/csv_handler/violation_tester")


def display_violations(violation_list, message_to_print, pynever_column, second_column):
    if any(violation_list):
        print(message_to_print)
        temporary_df = pd.DataFrame()
        temporary_df["pynever"] = pynever_column
        temporary_df["broken_column"] = second_column
        print(temporary_df)


def print_violations(violation_list, message_to_print, pynever_column, second_column):

    if any(violation_list):
        temporary_df = pd.DataFrame()
        temporary_df["pynever"] = pynever_column
        temporary_df["broken_column"] = second_column
        violation_logger.debug(message_to_print)
        violation_logger.debug(str(temporary_df) + '\n')


class ViolationsManager:
    def __init__(self, path1, path2, path3):
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3

        self.pynever_csv = pd.read_csv(self.path1)
        self.gimelli_csv = pd.read_csv(self.path2)
        self.elena_csv = pd.read_csv(self.path3)

    def check(self):
        for index, column in enumerate(self.pynever_csv.columns):
            pynever_column = self.pynever_csv[column]
            elena_column = self.elena_csv[column]
            gimelli_column = self.gimelli_csv[column]

            if index % 2 == 0:
                # check fc_csv and elena_csv LOWER
                gimelli_violations_list = pynever_column[pynever_column.notna()] < gimelli_column[
                    gimelli_column.notna()]
                elena_violations_list = pynever_column[pynever_column.notna()] < elena_column[gimelli_column.notna()]

                msg1 = "violations.txt on gimelli_bounds_prop: " + str(column)
                print_violations(gimelli_violations_list, msg1, pynever_column, gimelli_column)
                msg2 = "violations.txt on elena_bounds_prop: " + str(column)
                print_violations(elena_violations_list, msg2, pynever_column, elena_column)

            else:
                gimelli_violations_list = pynever_column[pynever_column.notna()] > gimelli_column[
                    gimelli_column.notna()]
                elena_violations_list = pynever_column[pynever_column.notna()] > elena_column[gimelli_column.notna()]

                msg1 = "violations.txt on gimelli_bounds_prop: " + str(column)
                print_violations(gimelli_violations_list, msg1, pynever_column, gimelli_column)
                msg2 = "violations.txt on elena_bounds_prop: " + str(column)
                print_violations(elena_violations_list, msg2, pynever_column, elena_column)
