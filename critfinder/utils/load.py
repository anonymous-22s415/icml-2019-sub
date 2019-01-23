import json

import autograd.numpy as np

import critfinder.experiments
import critfinder.networks


def fetch_data(data_json_path):
    data_dict = open_json(data_json_path)
    xs = np.load(data_dict["data_path"])
    data = (xs, xs)
    return data


def open_json(jfn):
    with open(jfn) as jf:
        json_data = json.load(jf)
    return json_data


def from_paths(data_json_path, network_json_path, experiment_json_path,
               experiment_type="optimization"):
    data = fetch_data(data_json_path)

    network = critfinder.networks.FeedforwardNetwork.from_json(
        data, network_json_path)

    if experiment_type == "optimization":
        experiment = critfinder.experiments.OptimizationExperiment.from_json(
            network.loss, experiment_json_path)
    elif experiment_type == "critfinder":
        experiment = critfinder.experiments.CritFinderExperiment.from_json(
            network.loss, experiment_json_path)
    else:
        raise NotImplementedError("experiment_type {} not understood"
                                  .format(experiment_type))

    return data, network, experiment
