import autograd.numpy as np

import critfinder.utils


N = 16
K = 4
CP_CUTOFF = 1e-10
SPECTRUM_CUTOFF = 1e-5


def filter_to_candidate_cps(cp_df, network,
                            cp_cutoff=CP_CUTOFF,
                            cut_early=False,
                            return_failures=False):
    candidate_cp_df = cp_df.loc[cp_df.final_squared_grad_norm < cp_cutoff, :]
    failure_cp_df = cp_df.loc[cp_df.final_squared_grad_norm > cp_cutoff, :]

    if cut_early:
        out = [apply_cutoff(row, cp_cutoff) for _, row in candidate_cp_df.iterrows()]
        candidate_cp_df.thetas, candidate_cp_df.squared_grad_norms = \
            [thetas for thetas, _ in out], [squared_grad_norms for _, squared_grad_norms in out]

    candidate_cp_df["candidate_theta"] = [row.thetas[-1] for _, row in candidate_cp_df.iterrows()]
    candidate_cp_df["candidate_loss"] = [network.loss(candidate_theta)
                                         for candidate_theta in candidate_cp_df.candidate_theta]

    if return_failures:
        return candidate_cp_df, failure_cp_df
    else:
        return candidate_cp_df


def make_cp_dfs(cf_ids, cf_df):
    cp_dfs = [critfinder.utils.analyze.construct_cp_df(cf_df.loc[cf_id])
              for cf_id in cf_ids]

    for idx, cp_df in enumerate(cp_dfs):
        cp_df["finder_index"] = [idx] * len(cp_df)

    return cp_dfs


def apply_cutoff(cp_row, cp_cutoff):
    thetas, squared_grad_norms = cp_row.thetas, cp_row.squared_grad_norms

    above_cutoff = np.greater_equal(squared_grad_norms, cp_cutoff)

    if sum(above_cutoff) == 0:
        return thetas[:1, :, :], squared_grad_norms[:1]
    else:
        return thetas[above_cutoff], squared_grad_norms[above_cutoff]


def get_hessian_info(thetas, network, spectrum_cutoff=SPECTRUM_CUTOFF):
    hessians = [np.squeeze(network.hess(theta)) for theta in thetas]

    spectrums = [np.linalg.eigvalsh(hessian) for hessian in hessians]

    morse_indices = [compute_morse_index(spectrum) for spectrum in spectrums]

    return hessians, spectrums, morse_indices


def compute_morse_index(spectrum, eps=SPECTRUM_CUTOFF):
    return np.mean(np.less(spectrum, -eps))


def theta_to_map(theta, network):
    _map = weights_to_map(network.extract_weights(theta))
    return _map


def weights_to_map(weights):
    return np.dot(weights[1], weights[0])


def map_to_index_set(_map, threshold=0.9, max_set_size=K):
    thresholded_map = np.where(_map > threshold, _map, 0.0)
    map_diagonal = np.diag(thresholded_map)
    assert np.array_equal(np.diag(map_diagonal), thresholded_map)

    nonzeros = np.nonzero(map_diagonal)[0]
    assert len(nonzeros) <= max_set_size

    index_set = tuple(nonzeros)
    return index_set


def compute_marginal_counts(counts, n=N):
    marginal_counts = []
    for ii in range(n):
        marginal_counts.append(sum([count for index_set, count in counts.items()
                                    if ii in index_set]))
    return marginal_counts


def log(ps):
    logs = np.log2(ps)
    logs[np.isinf(logs)] = 0
    return logs


def entropy(ps):
    return np.dot(ps, -log(ps))


def compute_entropy(cp_df):

    count = cp_df.groupby("index_set").count().final_loss

    marginal_count = [count.iloc[0]] + compute_marginal_counts(count)

    marginal_count /= np.sum(marginal_count)

    return entropy(marginal_count)


def bootstrap_entropy_sd(cp_df, num_bootstraps=100):
    entropies = []
    for _ in range(num_bootstraps):
        entropies.append(compute_entropy(cp_df.sample(len(cp_df), replace=True)))

    return np.std(entropies)
