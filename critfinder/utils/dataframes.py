import os

import pandas as pd

from . import load


def construct_experiments_df(experiments_path):
    experiments_path = os.path.abspath(experiments_path)
    experiments_path_elems = [os.path.join(experiments_path, elem)
                              for elem in os.listdir(experiments_path)]
    experiment_paths = [elem for elem in experiments_path_elems
                        if os.path.isdir(elem) and is_experiment_dir(elem)]

    experiment_IDs = [os.path.basename(experiment_path)
                      for experiment_path in experiment_paths]

    rows = [construct_experiment_row(experiment_path)
            for experiment_path in experiment_paths]

    return pd.DataFrame(data=rows, index=experiment_IDs)


def construct_experiment_row(experiment_path):
    json_names, json_files = zip(*[("".join(elem.split(".")[:-1]), elem)
                                 for elem in os.listdir(experiment_path)
                                 if elem.endswith(".json")])
    json_paths = [os.path.join(experiment_path, json_file) for json_file in json_files]

    json_dicts = [load.open_json(json_path) for json_path in json_paths]

    experiment_row = {}
    for json_dict in json_dicts:
        experiment_row.update(json_dict)

    for json_name, json_path in zip(json_names, json_paths):
        experiment_row[json_name + "_json"] = json_path

    return experiment_row


def is_experiment_dir(dir):
    """quick and dirty check"""
    return any([elem.endswith(".json") for elem in os.listdir(dir)])


def reconstruct_from_row(experiment_row, experiment_type="optimization"):

    if experiment_type not in ["optimization", "critfinder"]:
        raise NotImplementedError("experiment_type {} not understood"
                                  .format(experiment_type))

    if experiment_type == "critfinder":
        experiment_json_path = experiment_row.finder_json
        optimization_path = experiment_row.optimization_path
        optimization_row = pd.Series(construct_experiment_row(optimization_path))
    else:
        experiment_json_path = experiment_row.optimizer_json
        optimization_row = experiment_row

    data_json_path = optimization_row.data_json
    network_json_path = optimization_row.network_json

    data, network, experiment = load.from_paths(
        data_json_path, network_json_path, experiment_json_path,
        experiment_type=experiment_type)

    return data, network, experiment
