import argparse
import os

import autograd.numpy as np

import critfinder
import critfinder.utils.load


def main(args):
    experiment_json_path = fetch_experiment_json_path(args.critfinder_path)
    experiment_dict = critfinder.utils.load.open_json(experiment_json_path)

    data_json_path, network_json_path = fetch_json_paths(
        experiment_dict["optimization_path"])

    trajectory = fetch_trajectory(experiment_dict["trajectory_path"])

    finder_json_path = experiment_dict["finder_json"]

    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    _, _, finder_experiment = critfinder.utils.load.from_paths(
        data_json_path, network_json_path, finder_json_path,
        experiment_type="critfinder")

    init_theta = initialize_theta(
        finder_experiment, experiment_dict["init_theta"], trajectory,
        experiment_dict["theta_perturb"])

    finder_experiment.run(init_theta, args.num_iters)

    finder_experiment.save_results(output_path)


def initialize_theta(experiment, init_theta_str, thetas, theta_perturb):
    if init_theta_str == "uniform":
        init_theta = experiment.uniform(thetas)
    elif init_theta_str == "uniform_f":
        init_theta = experiment.uniform_f(thetas)
    elif init_theta_str.endswith(".npy"):
        init_theta = np.load(init_theta_str)
    else:
        raise NotImplementedError("init_theta_str {} not understood"
                                  .format(init_theta_str))

    if theta_perturb is not None:
        perturb_stdev = np.sqrt(10 ** theta_perturb)
        init_theta += perturb_stdev * np.random.standard_normal(size=init_theta.shape)

    return init_theta


def fetch_experiment_json_path(critfinder_path):
    assert "experiment.json" in os.listdir(critfinder_path)

    return os.path.abspath(os.path.join(critfinder_path, "experiment.json"))


def fetch_json_paths(optimization_path):
    json_filenames = ["data.json", "network.json"]
    optimization_dir_contents = os.listdir(optimization_path)

    assert all([json_filename in optimization_dir_contents
               for json_filename in json_filenames])
    return [os.path.abspath(os.path.join(optimization_path, json_filename))
            for json_filename in json_filenames]


def fetch_trajectory(trajectory_path):
    if trajectory_path.endswith(".npz"):
        results_npz = np.load(trajectory_path)
        trajectory = results_npz["theta"]
    else:
        raise NotImplementedError("trajectory_path {} not understood"
                                  .format(trajectory_path))
    return trajectory


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Load and run a critfinder experiment from its constituent files.")

    parser.add_argument("critfinder_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("num_iters", type=int)

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    main(parser.parse_args())
