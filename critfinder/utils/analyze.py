import os

import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR_NAME = "output"


def construct_cp_df(critfinder_row):

    finder_kwargs = critfinder_row.finder_kwargs

    finder_dir = os.path.dirname(critfinder_row.finder_json)
    output_dir = os.path.join(finder_dir, OUTPUT_DIR_NAME)

    output_npzs = [np.load(os.path.join(output_dir, elem))
                   for elem in os.listdir(output_dir)
                   if elem.endswith("npz")]

    row_dictionaries = []
    for output_npz in output_npzs:
        row_dictionary = {}

        row_dictionary.update(finder_kwargs)

        if "theta" in output_npz.keys():
            row_dictionary["thetas"] = output_npz["theta"]
            row_dictionary["final_theta"] = row_dictionary["thetas"][-1]
            row_dictionary["run_length"] = len(row_dictionary["thetas"])

        if "f_theta" in output_npz.keys():
            row_dictionary["losses"] = output_npz["f_theta"]
            row_dictionary["final_loss"] = row_dictionary["losses"][-1]

        if "g_theta" in output_npz.keys():
            row_dictionary["squared_grad_norms"] = 2 * output_npz["g_theta"]
            row_dictionary["final_squared_grad_norm"] = row_dictionary["squared_grad_norms"][-1]

        row_dictionaries.append(row_dictionary)

    return pd.DataFrame(row_dictionaries)


def plot_trajectories(cp_df, key, plot_func="plot", func=lambda x: x, ax=None,
                      subplots_kwargs=None, plot_func_kwargs=None):

    if ax is None:
        if subplots_kwargs is None:
            subplots_kwargs = {}

        f, ax = plt.subplots(**subplots_kwargs)

    if plot_func_kwargs is None:
        plot_func_kwargs = {}

    for ii, row in cp_df.iterrows():
        ys = func(row[key])
        if plot_func == "plot":
            ax.plot(ys, **plot_func_kwargs)

    return ax


def compute_maps(weight_lists):
    maps = []
    for weight_list in weight_lists:
        # could be more generically implemented as rfold with dot
        maps.append(np.dot(weight_list[1], weight_list[0]))
    return maps


def extract_weight_lists(cp_df, network, index=-1):
    weight_lists = []
    for _, row in cp_df.iterrows():
        weight_lists.append(network.extract_weights(row["thetas"][index]))
    return weight_lists
