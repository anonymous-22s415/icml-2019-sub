import argparse
import json
import os
import sys

import autograd.numpy as np

import critfinder.experiments

import critfinder.utils.util


DEFAULT_VERBOSITY = 0

DEFAULT_INIT_THETA = "uniform"
DEFAULT_LOG_KWARGS = {"track_theta": True,
                      "track_g": True,
                      "track_f": True}

DEFAULT_ALPHA = critfinder.finders.gradnormmin.DEFAULT_ALPHA
DEFAULT_BETA = critfinder.finders.gradnormmin.DEFAULT_BETA
DEFAULT_RHO = critfinder.finders.gradnormmin.DEFAULT_RHO
DEFAULT_GAMMA = critfinder.finders.gradnormmin.DEFAULT_GAMMA

DEFAULT_RTOL = 1e-10
DEFAULT_MAXIT = 25

DEFAULT_NEWTON_STEP_SIZE = DEFAULT_ALPHA

DEFAULT_TR_GAMMA_MX, DEFAULT_TR_GAMMA_K = 20, 30

DEFAULT_GNM_MINIMIZER = "btls"
DEFAULT_GNM_MINIMIZER_PARAMS = {"alpha": DEFAULT_ALPHA,
                                "beta": DEFAULT_BETA}
DEFAULT_GNM_CRITERION = "wolfe"
DEFAULT_GNM_CRITERION_PARAMS = {"rho": DEFAULT_RHO,
                                "gamma": DEFAULT_GAMMA}

DEFAULT_LR = DEFAULT_ALPHA
DEFAULT_MOMENTUM = 0.1


def main(args):

    critfinder_path, finder_json_path = extract_and_set_up_paths(args)

    finder_str, finder_kwargs = extract_finder_info(args)
    finder_experiment = critfinder.CritFinderExperiment(lambda x: None, finder_str, finder_kwargs)
    finder_experiment.to_json(finder_json_path)

    experiment_dict, experiment_json_path = setup_experiment_json(
        args, critfinder_path, finder_json_path)
    write_experiment_json(experiment_dict, experiment_json_path)


def extract_and_set_up_paths(args):

    ID = args.ID or critfinder.utils.util.random_string(6)

    # optimization_name = os.path.basename(args.optimization_path.rstrip("/"))
    critfinder_path = os.path.join(args.base_critfinders_path, ID)

    finder_json_path = args.finder_json_path or os.path.join(critfinder_path, "finder.json")

    os.makedirs(critfinder_path, exist_ok=True)

    return critfinder_path, finder_json_path


def setup_experiment_json(args, critfinder_path, finder_json_path):

    ID = os.path.basename(critfinder_path)

    optimization_path = args.optimization_path

    default_trajectories_path = os.path.join(optimization_path, "trajectories")
    trajectories_path = args.trajectories_path or default_trajectories_path
    trajectory_path = os.path.join(trajectories_path, args.trajectory_file)

    default_experiment_json_path = os.path.join(critfinder_path, "experiment.json")
    experiment_json_path = args.experiment_json_path or default_experiment_json_path

    experiment_dict = {"optimization_path": os.path.abspath(optimization_path),
                       "ID": ID,
                       "trajectory_path": os.path.abspath(trajectory_path),
                       "init_theta": args.init_theta,
                       "theta_perturb": args.theta_perturb,
                       "finder_json": os.path.abspath(finder_json_path)
                       }

    return experiment_dict, experiment_json_path


def write_experiment_json(experiment_dict, experiment_json_path):
    with open(experiment_json_path, "w") as fp:
        json.dump(experiment_dict, fp)


def extract_finder_info(args):
    finder_str = args.finder
    finder_kwargs = {"log_kwargs": DEFAULT_LOG_KWARGS}

    if finder_str.startswith("newton"):
        finder_kwargs.update(extract_newton_kwargs(args))
        if finder_str.endswith("MR"):
            finder_kwargs.update(extract_mr_kwargs(args))
        elif finder_str.endswith("TR"):
            finder_kwargs.update(extract_tr_kwargs(args))
        else:
            raise NotImplementedError("finder_str {0} not understood".format(finder_str))
    elif finder_str == "gnm":
        finder_kwargs.update(extract_gnm_kwargs(args))
    else:
        raise NotImplementedError("finder_str {0} not understood".format(finder_str))
    return finder_str, finder_kwargs


def extract_newton_kwargs(args):
    newton_kwargs = {}
    return newton_kwargs


def extract_tr_kwargs(args):
    tr_kwargs = {"step_size": args.newton_step_size,
                 "gammas": construct_gammas(args)}
    tr_kwargs.update(extract_minresqlp_kwargs(args))
    return tr_kwargs


def construct_gammas(args):
    mx, k = args.gamma_mx, args.gamma_k
    gammas = np.logspace(num=k, start=mx, stop=mx - k, endpoint=False).tolist()
    return gammas


def extract_mr_kwargs(args):
    mr_kwargs = {"alpha": args.alpha,
                 "beta": args.beta,
                 "rho": args.rho}
    mr_kwargs.update(extract_minresqlp_kwargs(args))
    return mr_kwargs


def extract_minresqlp_kwargs(args):
    return {"rtol": args.rtol, "maxit": args.maxit}


def extract_gnm_kwargs(args):
    gnm_kwargs = {"minimizer_str": args.minimizer}
    gnm_kwargs["minimizer_params"] = extract_minimizer_params(args)

    criterion_str, criterion_params = extract_criterion_info(args)
    gnm_kwargs["criterion_str"] = criterion_str
    gnm_kwargs["criterion_params"] = criterion_params

    return gnm_kwargs


def extract_minimizer_params(args):
    minimizer_params = {}
    if args.minimizer in ["gd", "momentum"]:
        minimizer_params["lr"] = args.lr
        if args.minimizer == "momentum":
            minimizer_params["momentum"] = args.momentum
    elif args.minimizer == "btls":
        minimizer_params["alpha"] = args.alpha
        minimizer_params["beta"] = args.beta
    else:
        raise NotImplementedError("minimizer_str {0} not understood"
                                  .format(args.minimizer))

    return minimizer_params


def extract_criterion_info(args):

    if args.criterion == "none":
        criterion_str = None
        criterion_params = None
    elif args.criterion in ["wolfe", "roosta"]:
        criterion_str = args.criterion
        criterion_params = {"gamma": args.gamma,
                            "rho": args.rho}
    else:
        raise NotImplementedError("criterion_str {0} not understood"
                                  .format(args.criterion))

    return criterion_str, criterion_params


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Creates the necessary files to run a critfinder experiment.")

    # PROGRAM
    parser.add_argument("-v",
                        action="store_const", dest="verbosity", const=1, default=DEFAULT_VERBOSITY,
                        help="verbosity flag")

    # EXPERIMENT
    parser.add_argument("optimization_path", type=str,
                        help="path to base optimization.")
    parser.add_argument("--base_critfinders_path", type=str, default="results/critfinders",
                        help="path to critfinders folder, in which critfinders on optimizations " +
                        "are stored in folders.")
    parser.add_argument("--trajectories_path", type=str, default=None,
                        help="provide to override default behavior, " +
                        "which is to use {optimization_path}/trajectories.")
    parser.add_argument("--trajectory_file", type=str, default="0000.npz",
                        help="basename of trajectory file within {trajectories_path}, " +
                        "default is 0000.npz.")
    parser.add_argument("--init_theta", type=str, default=DEFAULT_INIT_THETA,
                        help="either a string identifying an initialization strategy " +
                        "based on a trajectory or a string path to an init_theta npy file. " +
                        "initialization strategies are {uniform, uniform_f} for selecting " +
                        "points uniformly from the trajectory or " +
                        "with a uniform distribution on heights.")
    parser.add_argument("--theta_perturb", type=float, default=None,
                        help="Logarithm base 10 of amount to perturb theta" +
                        "initialized from optimization trajectory." +
                        "sets variance of additive gaussian noise." +
                        "default is None, which results in no perturbation.")
    parser.add_argument("--finder_json_path", type=str, default=None,
                        help="provide to override default behavior, " +
                        "which is to place finder.json in critfinder folder.")
    parser.add_argument("--experiment_json_path", type=str, default=None,
                        help="provide to override default behavior, " +
                        "which is to place experiment.json in critfinder folder.")
    parser.add_argument("--ID", type=str, default=None,
                        help="if not provided, a random string is generated as ID.")

    # CRITFINDER
    parser.add_argument("finder", type=str, choices=["gnm", "newtonMR", "newtonTR"],
                        help="string identifying the finder to use.")

    # btls
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="used by all BTLS. " +
                        "initial value for step size. " +
                        "default is {}".format(DEFAULT_ALPHA))
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA,
                        help="used by all BTLS. " +
                        "multiplicative factor by which to decrease step size on " +
                        "each failed line search step. " +
                        "default is {}".format(DEFAULT_BETA))
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO,
                        help="used in armijo check in all BTLS. " +
                        "hyperparameter for strictness of sufficient decrease condition. " +
                        "default is {}".format(DEFAULT_RHO))

    # gnm
    parser.add_argument("--minimizer", type=str, default=DEFAULT_GNM_MINIMIZER,
                        help="minimizer to use on gnm. " +
                        "default is {}".format(DEFAULT_GNM_MINIMIZER))
    parser.add_argument("--criterion", type=str, default=DEFAULT_GNM_CRITERION,
                        help="stopping criterion to use on btls for gnm. " +
                        "default is {}".format(DEFAULT_GNM_CRITERION))
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help="used in wolfe criterion on gnm. " +
                        "hyperparameter for strictness of curvature condition. " +
                        "default is {}".format(DEFAULT_GAMMA))

    # gd and momentum
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="used by gnm. " +
                        "learning rate for gradient descent or momentum minimizer. " +
                        "default is {}".format(DEFAULT_LR))
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM,
                        help="used by gnm. " +
                        "fraction of momentum term to preserve across steps. "
                        "default is {}".format(DEFAULT_MOMENTUM))

    # minresqlp
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--maxit", type=int, default=DEFAULT_MAXIT)

    # newton-tr
    parser.add_argument("--newton_step_size", type=float, default=DEFAULT_NEWTON_STEP_SIZE,
                        help="used by non-BTLS Newtons. " +
                        "step size for Newton method. " +
                        "default is {}".format(DEFAULT_NEWTON_STEP_SIZE))
    parser.add_argument("--gamma_mx", type=float, default=DEFAULT_TR_GAMMA_MX,
                        help="used by trust-region Newton. " +
                        "maximum order of magnitude for trust region size parameter. " +
                        "default is {}".format(DEFAULT_TR_GAMMA_MX))
    parser.add_argument("--gamma_k", type=int, default=DEFAULT_TR_GAMMA_K,
                        help="used by trust-region Newton. " +
                        "number of orders of magnitude over which to vary " +
                        "trust region size parameter. " +
                        "default is {}".format(DEFAULT_TR_GAMMA_K))

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    main(parser.parse_args())
    sys.exit(0)
