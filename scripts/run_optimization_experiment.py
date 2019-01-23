import argparse
from collections import namedtuple
import os
import sys

import autograd.numpy as np

import critfinder.utils.load

DEFAULT_SUFFIXES = ["data.json", "network.json", "optimizer.json", "trajectories/0000.npz"]
PATH_NAMES = ["optimization", "data", "network", "optimizer", "trajectory"]

BIAS_EPS = 0.01

Paths = namedtuple("Paths", PATH_NAMES)


def main(num_steps, paths, init_theta=None):

    _, network, optimization_experiment = critfinder.utils.load.from_paths(
        paths.data, paths.network, paths.optimizer)

    init_theta = initialize_theta(init_theta, network)

    optimization_experiment.run(init_theta, num_steps)

    optimization_experiment.save_results(paths.trajectory)


def setup_paths(optimization_path, data_path, network_path, optimizer_path, trajectory_path):

    path_list = [data_path, network_path, optimizer_path, trajectory_path]

    if any([path is None for path in path_list]):
        assert optimization_path is not None

    processed_path_list = []
    for path, default_suffix in zip(path_list, DEFAULT_SUFFIXES):
        if path is None:
            path = os.path.join(optimization_path, default_suffix)
        processed_path_list.append(path)

    paths = Paths(optimization_path, *processed_path_list)

    return paths


def initialize_theta(init_theta, network):
    # TODO: make initialization network's job. move init to setup, save to file?
    if init_theta is None:
        k = network.layers[0]
        p = network.layers[1]
        init_theta = 1 / np.sqrt(k * p) * np.atleast_2d(np.random.standard_normal(size=k * p * 2))

        if network.has_biases:
            biases = np.atleast_2d([BIAS_EPS] * (p + k))
            init_theta = np.hstack([init_theta, biases])

        init_theta = init_theta.T

    elif isinstance(init_theta, str):
        init_theta = np.load(init_theta)

    return init_theta


def construct_parser():

    parser = argparse.ArgumentParser(
        description='Execute an optimization experiment and save the resulting trajectory.')

    parser.add_argument("num_steps", type=int,
                        help="Number of steps of optimizer to apply.")

    parser.add_argument("--optimization_path", type=str, default=None,
                        help="path to directory containing .json files describing optimization, " +
                        "defaults to None, which only works if all other _path arguments are " +
                        "provided explicitly.")

    parser.add_argument("--init_theta", type=str, default=None,
                        help="path to initial parameter values")

    for default_suffix, path_name in zip(DEFAULT_SUFFIXES, PATH_NAMES[1:]):

        parser.add_argument("--{}_path".format(path_name), type=str, default=None,
                            help="path to {}. ".format(path_name) +
                            "default is inside optimization_path at {}"
                            .format(default_suffix))

    return parser


if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    paths = setup_paths(
        args.optimization_path,
        args.data_path, args.network_path, args.optimizer_path, args.trajectory_path)
    main(args.num_steps, paths, args.init_theta)
    sys.exit(0)
