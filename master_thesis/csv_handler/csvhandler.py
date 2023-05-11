import pandas as pd
import csv
import numpy as np
from itertools import zip_longest
import math
from pynever.strategies.bound_propagation_gimelli.bounds_menager import MyBoundsManager
from pynever.strategies.bound_propagation_elena.verification.bounds.boundsmanagerelena import BoundsManagerElena


class Bounds:
    def __init__(self):
        self.lower = None
        self.upper = None


def print_multiple_csv(abst_network, prop, bound_propagation_type, path1=None, path2=None):
    if bound_propagation_type == 0:
        pass
    if bound_propagation_type == 1:
        gimelli_bounds_menager = MyBoundsManager(abst_network, prop)
        gimelli_symb_bounds, gimelli_numeric_bounds = gimelli_bounds_menager.compute_bounds()
        print_to_csv(gimelli_numeric_bounds, path1)

    elif bound_propagation_type == 2:
        elena_bounds_menager = BoundsManagerElena(abst_network, prop)
        elena_symb_bounds, elena_numeric_bounds = elena_bounds_menager.compute_bounds()
        print_to_csv(elena_numeric_bounds, path2)

    elif bound_propagation_type == 3:
        gimelli_bounds_menager = MyBoundsManager(abst_network, prop)
        gimelli_symb_bounds, gimelli_numeric_bounds = gimelli_bounds_menager.compute_bounds()
        elena_bounds_menager = BoundsManagerElena(abst_network, prop)
        elena_symb_bounds, elena_numeric_bounds, post_bounds = elena_bounds_menager.compute_bounds()
        print_to_csv(gimelli_numeric_bounds, path1)
        print_to_csv(elena_numeric_bounds, path2)
        print_to_csv(post_bounds, path2.replace(".csv", "_symb.csv"))


def fill_with_nan(lst, length):
    result = lst.copy()
    while len(result) < length:
        result.append(math.nan)
    return result


def remove_empty_string(bounds_list):
    return [x for x in bounds_list if x != '']


def from_string_list_to_float(bounds_list):
    return [list(map(float, x.strip('[]').split(','))) for x in bounds_list]


def print_to_csv(data_dict: dict, output_file_path):
    to_print = list()
    header = list()
    length_list = list()
    for key in data_dict.keys():
        key_lower = key + "_lower"
        key_upper = key + "_upper"
        header.append(key_lower)
        header.append(key_upper)
        to_print.append(data_dict[key].lower.tolist())
        to_print.append(data_dict[key].upper.tolist())
        length_list.append(len(data_dict[key].lower.tolist()))

    max_length = max(length_list)
    for i in range(len(to_print)):
        to_print[i] = fill_with_nan(to_print[i], max_length)

    with open(output_file_path, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in zip(*to_print):
            writer.writerow(row)


def print_to_csv_pynever_bounds(input_file_path, output_file_path):
    file = open(input_file_path, 'r').read()
    list_of_rows = file.split('\n')

    list_of_labels = list()
    list_of_lower_bounds = list()
    list_of_upper_bounds = list()

    for i, element in enumerate(list_of_rows):
        if i % 3 == 0:
            if element != "":
                lower_label = element.split('&')[0]
                upper_label = element.split('&')[1]
                list_of_labels.append(lower_label)
                list_of_labels.append(upper_label)
        elif i % 3 == 1:
            list_of_lower_bounds.append(element)
        else:
            list_of_upper_bounds.append(element)

    list_of_labels = [x for x in list_of_labels if x != '']
    list_of_lower_bounds = remove_empty_string(list_of_lower_bounds)
    list_of_upper_bounds = remove_empty_string(list_of_upper_bounds)

    list_of_lower_bounds = from_string_list_to_float(list_of_lower_bounds)
    list_of_upper_bounds = from_string_list_to_float(list_of_upper_bounds)

    max_len = len(max(list_of_upper_bounds, key=len))

    list_of_lower_bounds = [fill_with_nan(x, max_len) for x in list_of_lower_bounds]
    list_of_upper_bounds = [fill_with_nan(x, max_len) for x in list_of_upper_bounds]
    to_print = [x for pair in zip(list_of_lower_bounds, list_of_upper_bounds) for x in pair]

    with open(output_file_path, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list_of_labels)
        for row in zip(*to_print):
            writer.writerow(row)


def get_lower_upper(stars):
    lower_list_of_lists = list()
    upper_list_of_lists = list()

    for star in stars:
        lower = list()
        upper = list()
        for i in range(star.center.shape[0]):
            lb, ub = star.get_bounds(i)
            #print('i'+ str(lb)+" "+str(ub))
            lower.append(lb)
            upper.append(ub)
        lower_list_of_lists.append(lower)
        upper_list_of_lists.append(upper)

    absolute_lower_bounds = [min(x) for x in zip(*lower_list_of_lists)]
    absolute_upper_bounds = [max(x) for x in zip(*upper_list_of_lists)]

    return absolute_lower_bounds, absolute_upper_bounds


def from_stars_to_csv(stars_dict: dict):
    layer_bounds_dict = dict()
    all_layer_lower_bounds = list()
    all_layer_upper_bounds = list()

    counter = 1
    for key, layer in stars_dict.items():
        lower, upper = get_lower_upper(layer.stars)
        all_layer_lower_bounds.append(lower)
        all_layer_upper_bounds.append(upper)

        lower_key = key + "_lower"
        upper_key = key + "_upper"

        layer_bounds_dict[lower_key] = lower
        layer_bounds_dict[upper_key] = upper

        counter += 1

    return layer_bounds_dict

