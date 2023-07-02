import pandas as pd
import logging
from pynever.strategies.bound_propagation_alternative.bounds_manager import *
from pynever.strategies.bound_propagation.verification.bounds.bounds_manager import *
from pynever.strategies import smt_reading, verification
from csv_generator.csvhandler import print_to_csv_pynever_bounds, print_to_csv, from_stars_to_csv
import pynever.strategies.abstraction as abst
import math

violation_logger = logging.getLogger("pynever/master_thesis/csv_handler/violation_tester")


def getAbstractNetwork(net):
    fully_connected_counter = 0
    relu_counter = 0

    # ABSTRACT SEQUENTIAL NETWORK
    abs_net = abst.AbsSeqNetwork("SmallAbstractNetwork")

    node = net.get_first_node()

    if isinstance(node, nodes.FullyConnectedNode):
        title = "AbsFC_" + str(fully_connected_counter)
        abs_net.add_node(abst.AbsFullyConnectedNode(title, node))
        fully_connected_counter += 1
    elif isinstance(node, nodes.ReLUNode):
        title = "AbsReLU_" + str(relu_counter)
        abs_net.add_node(abst.AbsReLUNode(title, node, "best_n_neurons", [10]))
        relu_counter += 1

    while node is not net.get_last_node():

        node = net.get_next_node(node)

        if isinstance(node, nodes.FullyConnectedNode):
            title = "AbsFC_" + str(fully_connected_counter)
            abs_net.add_node(abst.AbsFullyConnectedNode(title, node))
            fully_connected_counter += 1
        elif isinstance(node, nodes.ReLUNode):
            title = "AbsReLU_" + str(relu_counter)
            abs_net.add_node(abst.AbsReLUNode(title, node, "best_n_neurons", [1000]))
            relu_counter += 1

    # create layers list of abstract nodes
    layers = []

    abs_node = abs_net.get_first_node()
    layers.append(abs_node)

    while abs_node is not abs_net.get_last_node():
        abs_node = abs_net.get_next_node(abs_node)
        layers.append(abs_node)

    return abs_net


def print_violations(violation_list, message_to_print, pynever_column, second_column, on_log):
    if any(violation_list):
        temporary_df = pd.DataFrame()
        temporary_df["pynever"] = pynever_column
        temporary_df["broken_column"] = second_column

        if on_log:
            violation_logger.debug(message_to_print)
            violation_logger.debug(str(temporary_df) + '\n')
        else:
            print(message_to_print)
            print(temporary_df)


def print_lower_violations(pynever_column, second_column):
    for i in range(len(pynever_column)):
        if pynever_column[i] < second_column[i] and math.abs(pynever_column[i] - second_column[i]) < 0.001:
            print("big violation")


class ViolationsManager:
    def __init__(self, path1, path2, path3, net, prop, starts_dict):

        # path for PyNeVer stars bounds csv file
        self.path1 = path1

        # path for gimelli bounds csv file
        self.path2 = path2

        # path for elena bounds csv file
        self.path3 = path3

        bound_manager_1 = BoundsManager_1(getAbstractNetwork(net), prop)
        _, numeric_bounds_1 = bound_manager_1.compute_bounds()

        bound_manager_2 = BoundsManager_2(getAbstractNetwork(net), prop)
        _, numeric_bounds_2, post_bounds_2 = bound_manager_2.compute_bounds()

        # print bounds on csv file
        from_stars_to_csv(starts_dict, path1)
        print_to_csv(numeric_bounds_1, path2)
        print_to_csv(post_bounds_2, path3)

        self.pynever_csv = pd.read_csv(self.path1)
        self.csv_1 = pd.read_csv(self.path2)
        self.csv_2 = pd.read_csv(self.path3)

    def check(self, soglia, on_log=False):
        self.pynever_csv.columns = self.csv_1.columns

        for index, column in enumerate(self.pynever_csv.columns):
            pynever_column = self.pynever_csv[column]
            elena_column = self.csv_2[column]
            gimelli_column = self.csv_1[column]

            if index % 2 == 0:
                # check fc_csv and csv_2 LOWER
                gimelli_violations_list = pynever_column[pynever_column.notna()] < gimelli_column[
                    gimelli_column.notna()]

                elena_violations_list = pynever_column[pynever_column.notna()] < \
                                        elena_column[gimelli_column.notna()]

                msg1 = "violations.txt on gimelli_bounds_prop: " + str(column)
                print_violations(gimelli_violations_list, msg1, pynever_column, gimelli_column, on_log)
                msg2 = "violations.txt on elena_bounds_prop: " + str(column)
                print_violations(elena_violations_list, msg2, pynever_column, elena_column, on_log)

            else:
                gimelli_violations_list = pynever_column[pynever_column.notna()] > gimelli_column[
                    gimelli_column.notna()]

                elena_violations_list = pynever_column[pynever_column.notna()] > elena_column[
                    gimelli_column.notna()]

                msg1 = "violations.txt on gimelli_bounds_prop: " + str(column)
                print_violations(gimelli_violations_list, msg1, pynever_column, gimelli_column, on_log)
                msg2 = "violations.txt on elena_bounds_prop: " + str(column)
                print_violations(elena_violations_list, msg2, pynever_column, elena_column, on_log)
