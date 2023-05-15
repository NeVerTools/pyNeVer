import pandas as pd
import logging
from pynever.strategies.bound_propagation_gimelli.bounds_menager import *
from pynever.strategies.bound_propagation_elena.verification.bounds.boundsmanagerelena import *
from pynever.strategies import smt_reading, verification
from master_thesis.csv_handler.csvhandler import print_to_csv_pynever_bounds, print_to_csv, from_stars_to_csv
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

    # for i in range(len(pynever_column)):
    #     if pynever_column[i] < second_column[i] and abs(pynever_column[i] - second_column[i]) > 0.000001:
    #         print("big violation")

# def print_violations2(violation_list, message_to_print, pynever_column, second_column, on_log):
#     # if any(violation_list):
#     #     temporary_df = pd.DataFrame()
#     #     temporary_df["pynever"] = pynever_column
#     #     temporary_df["broken_column"] = second_column
#     #
#     #     if on_log:
#     #         violation_logger.debug(message_to_print)
#     #         violation_logger.debug(str(temporary_df) + '\n')
#     #     else:
#     #         print(message_to_print)
#     #         print(temporary_df)
#
#     for i in range(len(pynever_column)):
#         #if pynever_column[i] > second_column[i] and abs(pynever_column[i] - second_column[i]) > 0.000001:
#         if pynever_column[i] > second_column[i]:
#
#             print("big violation")

def print_lower_violations(pynever_column, second_column):
    for i in range(len(pynever_column)):
        if pynever_column[i] < second_column[i] and math.abs(pynever_column[i] - second_column[i]) < 0.001:
            print("big violation")


class ViolationsManager:
    def __init__(self, path1, path2, path3, net, prop, starts_dict):
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3

        bound_manager_gimelli = MyBoundsManager(getAbstractNetwork(net), prop)
        _, gimelli_numeric_bounds = bound_manager_gimelli.compute_bounds()

        bound_manager_elena = BoundsManagerElena(getAbstractNetwork(net), prop)
        _, elena_numeric_bounds, elena_post_bounds = bound_manager_elena.compute_bounds()

        # print bounds on csv file
        from_stars_to_csv(starts_dict, path1)
        print_to_csv(gimelli_numeric_bounds, path2)
        print_to_csv(elena_post_bounds, path3)

        self.pynever_csv = pd.read_csv(self.path1)
        self.gimelli_csv = pd.read_csv(self.path2)
        self.elena_csv = pd.read_csv(self.path3)

    def check(self, soglia, on_log=False):
        self.pynever_csv.columns = self.gimelli_csv.columns

        for index, column in enumerate(self.pynever_csv.columns):
            pynever_column = self.pynever_csv[column]
            elena_column = self.elena_csv[column]
            gimelli_column = self.gimelli_csv[column]

            if index % 2 == 0:
                # check fc_csv and elena_csv LOWER
                gimelli_violations_list = pynever_column[pynever_column.notna()] < gimelli_column[
                    gimelli_column.notna()]
                elena_violations_list = pynever_column[pynever_column.notna()]  < \
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
