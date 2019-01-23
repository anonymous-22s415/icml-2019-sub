import sys

import autograd.numpy as np
import itertools
import pandas as pd
import argparse

import critfinder


parser = argparse.ArgumentParser("Calculate critical point structure of a linear autoencoder.")

parser.add_argument("num_hidden", type=int)
parser.add_argument("data_path", type=str)
parser.add_argument("--save_to", type=str, default=None)
parser.add_argument("--stop_early", type=int, default=np.inf,
                    help="Maximum mumber of eigenvectors to include in a set. " +
                    "Defaults to allowing up to all eigenvectors.")


def main():
    args = parser.parse_args()

    num_hidden = args.num_hidden
    data_path = args.data_path
    stop_early = args.stop_early

    data = np.load(data_path)

    cp_df = make_cp_df(data, num_hidden, stop_early)

    if args.save_to is None:
        df_path = "analytical_cp_df.pkl"
    else:
        df_path = args.save_to

    save_cp_df(cp_df, df_path)

    return 0


def make_cp_df(data, num_hidden, stop_early=np.inf):
    num_inputs = data.shape[0]

    sigma = np.cov(data)
    data_evals, data_evecs = np.linalg.eigh(sigma)

    layers = [num_inputs, num_hidden, num_inputs]

    network = critfinder.FeedforwardNetwork((data, data), layers,
                                            nonlinearity_str="none", has_biases=False)

    n, p = num_inputs, num_hidden

    bh_costs = []
    costs = []
    cost_gradient_squared_norms = []

    ps = []
    ks = []
    index_sets = []
    index_set_booleans = []

    hessian_spectra = []

    for k in range(0, p + 1):

        if k > stop_early:
            break

        indices = range(n)
        k_index_sets = list(make_index_sets(k, indices))
        k_index_set_booleans = make_index_set_booleans(k_index_sets, n)

        for k_index_set_bool, k_index_set in zip(k_index_set_booleans, k_index_sets):
            bh_cost = np.trace(sigma) - np.sum(data_evals[k_index_set_bool])

            weights_as_vector = weights_from_index_set_bool(k_index_set_bool, data_evecs, p)

            cost = network.loss(weights_as_vector)
            cost_gradient_squared_norm = np.sum(np.square(network.grad(weights_as_vector)))
            hessian = np.squeeze(network.hess(weights_as_vector))
            hessian_spectrum = np.linalg.eigvalsh(hessian)

            bh_costs.append(bh_cost)
            costs.append(cost)
            cost_gradient_squared_norms.append(cost_gradient_squared_norm)
            ps.append(p)
            ks.append(k)
            index_sets.append(k_index_set)
            index_set_booleans.append(k_index_set_bool)
            hessian_spectra.append(hessian_spectrum)

    cp_df = pd.DataFrame.from_dict({"cost": costs,
                                    "bh_cost": bh_costs,
                                    "cost_gradient_squared_norm": cost_gradient_squared_norms,
                                    "p": ps,
                                    "k": ks,
                                    "index_set": index_sets,
                                    "index_set_boolean": index_set_booleans,
                                    "hessian_spectrum": hessian_spectra})
    return cp_df


def weights_from_index_set_bool(index_set_bool, data_evecs, p):
    n, k = len(index_set_bool), sum(index_set_bool)  # TEST THIS

    B = np.zeros((n, p))
    B[:, 0:k] = data_evecs[:, index_set_bool]
    A = np.transpose(B)

    weights_as_vector = np.atleast_2d(np.concatenate([A.ravel(), B.ravel()])).T

    return weights_as_vector


def subset_to_boolean(subset, cardinality):
    subset_boolean = np.zeros(cardinality, dtype=bool)
    for element in subset:
        subset_boolean[element] = 1
    return subset_boolean


def make_index_sets(index_set_cardinality, indices):
    index_sets = itertools.combinations(indices, index_set_cardinality)
    return index_sets


def make_index_set_booleans(index_sets, set_cardinality):
    return [subset_to_boolean(index_set, set_cardinality) for index_set in index_sets]


def save_cp_df(cp_df, filename):
    cp_df.to_pickle(filename)
    return


if __name__ == "__main__":
    sys.exit(main())
