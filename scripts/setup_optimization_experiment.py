import argparse
import json
import os
import sys

import autograd.numpy as np

import critfinder
from critfinder.utils.util import random_string

DEFAULT_VERBOSITY = 0

DEFAULT_RESULTS_PATH = os.path.join("results", "test")

DEFAULT_K = 16
DEFAULT_N = 10000

DEFAULT_P = 4
DEFAULT_REGULARIZER = "l2"
DEFAULT_REGULARIZATION_PARAMETER = 0.0
DEFAULT_NONLINEARITY = "none"

DEFAULT_OPTIMIZER = "gd"
DEFAULT_OPTIMIZER_LR = 0.1
DEFAULT_OPTIMIZER_MOMENTUM = 0.1

DEFAULT_LOG_KWARGS = {"track_theta": True, "track_f": True, "track_g": False}


def main(args):

    ID = args.ID or random_string(6)
    if args.verbosity > 0:
        print("creating files for {}".format(ID))

    network_path, optimizer_path, run_path = extract_and_setup_paths(args, ID)
    data, data_path = setup_data_and_data_path(args, run_path)
    optimizer_kwargs = extract_optimizer_kwargs(args)
    log_kwargs = setup_log_kwargs(args)

    MLP = critfinder.FeedforwardNetwork(
        data, layers=[args.k] + args.layers + [args.k], has_biases=args.has_biases,
        regularizer_str=args.regularizer, regularization_parameter=args.regularization_parameter,
        nonlinearity_str=args.nonlinearity)

    optimization_experiment = critfinder.OptimizationExperiment(
        MLP.loss,
        optimizer_str=args.optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_kwargs=log_kwargs)

    MLP.to_json(network_path)
    optimization_experiment.to_json(optimizer_path)
    write_data_path_to_json(data_path, run_path)

    if args.save_data:
        np.save(data_path, data[0])


def extract_and_setup_paths(args, ID):

    run_path = os.path.join(args.results_path, ID)
    trajectories_path = os.path.join(run_path, "trajectories")
    os.makedirs(trajectories_path, exist_ok=True)

    network_path = os.path.join(run_path, "network.json")
    optimizer_path = os.path.join(run_path, "optimizer.json")

    return network_path, optimizer_path, run_path


def setup_data_and_data_path(args, run_path):
    if args.save_data:
        Sigma = np.diag(range(1, args.k + 1))
        xs = np.random.multivariate_normal(mean=np.zeros(args.k), cov=Sigma, size=args.N).T

        if args.zero_centering == "subtract_mean":
            xs = xs - np.mean(xs, axis=1)[:, None]
        elif args.zero_centering == "add_point":
            xs = xs[:, :-1]
            xs = xs - np.mean(xs, axis=1)[:, None]
            approx_mu = np.mean(xs, axis=1)[:, None]
            xs = np.hstack([xs, -(args.N - 1) * approx_mu])

        data = (xs, xs)
        data_path = args.data_path
    else:
        data_path = args.data_path
        xs = np.load(data_path)
        data = (xs, xs)

    return data, data_path


def extract_optimizer_kwargs(args):
    optimizer_kwargs = {"lr": args.optimizer_lr}
    if args.optimizer == "momentum":
        if args.optimizer_momentum is not None:
            optimizer_kwargs["momentum"] = args.optimizer_momentum
        else:
            optimizer_kwargs["momentum"] = DEFAULT_OPTIMIZER_MOMENTUM

    return optimizer_kwargs


def setup_log_kwargs(args):
    log_kwargs = DEFAULT_LOG_KWARGS.copy()
    if args.log_gradients:
        log_kwargs["g_theta"] = True

    return log_kwargs


def write_data_path_to_json(data_path, run_path):
    data_json_path = os.path.abspath(os.path.join(run_path, "data.json"))
    data_abspath = os.path.abspath(data_path)

    with open(data_json_path, "w") as fn:
        json.dump({"data_path": data_abspath}, fn)


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Create necessary files for an optimization experiment.")

    # PROGRAM
    parser.add_argument("-v", dest="verbosity",
                        action="store_const", const=1, default=0,
                        help="verbosity flag")

    # PATHS
    parser.add_argument("--results_path", type=str, default=DEFAULT_RESULTS_PATH,
                        help="top-level directory for results. " +
                        "default is {}".format(DEFAULT_RESULTS_PATH))
    parser.add_argument("--ID", type=str, default=None,
                        help="identifier for this optimization problem." +
                        "provide to over-ride default behavior, generating a random ID.")

    # DATA
    parser.add_argument("--data_path", type=str, default=None,
                        help="path to data to load.")
    parser.add_argument("--N", type=int, default=DEFAULT_N,
                        help="number of data points to generate, if data_path not provided. " +
                        "default is {}".format(DEFAULT_N))
    parser.add_argument("--zero_centering", type=str, default="none",
                        choices=["none", "subtract_mean", "add_point"],
                        help="type of zero centering to apply to the data: " +
                        "subtract_mean, subtract the mean; " +
                        "add_point, attempt to add a number to the sample " +
                        "such that the data has mean exactly zero on each dimension; " +
                        "none, meaning do nothing. " +
                        "add_point is less subject to floating point error. " +
                        "Default is none.")
    parser.add_argument("--save_data", action="store_const", const=True, default=False)

    # NETWORK
    parser.add_argument("--layers", type=int, default=[DEFAULT_P], nargs="*",
                        help="sizes of hidden layers")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="number of input units. " +
                        "if data_path is provided, also sets size of data. " +
                        "default is {}".format(DEFAULT_K))
    parser.add_argument("--regularizer", type=str, default=DEFAULT_REGULARIZER,
                        choices=["none", "l1", "l2"],
                        help="type of regularization term to apply to weights. " +
                        "default is {}, but see --regularization_parameter"
                        .format(DEFAULT_REGULARIZER))
    parser.add_argument("--regularization_parameter", type=float,
                        default=DEFAULT_REGULARIZATION_PARAMETER,
                        help="multiplicative factor to apply to regularization cost. " +
                        "default is {}".format(DEFAULT_REGULARIZATION_PARAMETER))
    parser.add_argument("--nonlinearity", type=str, default=DEFAULT_NONLINEARITY,
                        help="nonlinear transform between layers. " +
                        "default is {}".format(DEFAULT_NONLINEARITY))
    parser.add_argument("--include_biases",
                        dest="has_biases", action="store_const", const=True, default=False,
                        help="flag to determine whether network has biases.")

    # OPTIMIZER
    parser.add_argument("--optimizer", type=str, default=DEFAULT_OPTIMIZER,
                        help="optimizer to apply to network loss. " +
                        "default is {}".format(DEFAULT_OPTIMIZER),
                        choices=["gd", "momentum"])
    parser.add_argument("--optimizer_lr", type=float, default=DEFAULT_OPTIMIZER_LR,
                        help="learning rate for optimizer. " +
                        "default is {}".format(DEFAULT_OPTIMIZER_LR))
    parser.add_argument("--optimizer_momentum", type=float, default=None,
                        help="momentum level for momentum optimizer. " +
                        "default is {}".format(DEFAULT_OPTIMIZER_MOMENTUM))
    parser.add_argument("--log_gradients",
                        dest="log_gradients", action="store_const", const=True, default=False,
                        help="flag to log gradients of loss with respect to theta" +
                        "during training.")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
    sys.exit(0)
