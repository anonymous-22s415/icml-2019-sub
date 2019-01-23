import autograd.numpy as np
import matplotlib.pyplot as plt

import panels
import utils


NUMERICAL_CPS_COLOR = panels.NUMERICAL_CPS_COLOR
ANALYTICAL_CPS_COLOR = panels.ANALYTICAL_CPS_COLOR


def make_performance_comparison_figure(cp_dfs, analytical_cp_df,
                                       failed_cp_dfs=None,
                                       titles=None,
                                       base_figure_scale=6):

    num_cp_dfs = len(cp_dfs)

    if titles is None:
        titles = [None] * num_cp_dfs

    if failed_cp_dfs is None:
        failed_cp_dfs = [None] * num_cp_dfs

    gridsize = [2, num_cp_dfs]
    figsize = base_figure_scale * np.asarray(gridsize[::-1])
    f, axs = plt.subplots(*gridsize, figsize=figsize)

    axs = np.atleast_2d(axs)

    for ii, (cp_df, failed_cp_df, title, ax_pair) in enumerate(
            zip(cp_dfs, failed_cp_dfs, titles, axs.T)):
        if ii == 0:
            include_y_label = True
        else:
            include_y_label = False

        include_legend = False
        if (ii + 1) == num_cp_dfs:
            include_legend = True

        panels.make_performance_panel_pair(
            cp_df, analytical_cp_df,
            title=title, axs=ax_pair,
            include_legend=include_legend,
            include_y_label=include_y_label,
            failed_cp_df=failed_cp_df)

    return f, axs


def make_cutoff_comparison_figure(cutoff_cp_dfs, cutoffs, analytical_cp_df,
                                  gridsize=[2, 2], base_figure_scale=3):

    figsize = base_figure_scale * np.asarray(gridsize)

    f, axs = plt.subplots(*gridsize, figsize=figsize,
                          sharex=True, sharey=True)

    include_x_label = include_y_label = True
    include_legend = False

    for ii, (cutoff_cp_df, cutoff, ax) in enumerate(
        zip(
            cutoff_cp_dfs, cutoffs, axs.flatten())):

        if ii // gridsize[0] == 0:
            include_x_label = False
        else:
            include_x_label = True
        if ii % gridsize[1] == 0:
            include_y_label = True
        else:
            include_y_label = False

        if ii + 1 == sum(gridsize):
            include_legend = True

        panels.make_loss_index_panel(
            cutoff_cp_df, analytical_cp_df,
            ax=ax,
            include_x_label=include_x_label,
            include_y_label=include_y_label,
            include_legend=include_legend)

        if cutoff < np.inf:
            cutoff_str = r"$\varepsilon={0}$".format(cutoff)
        else:
            cutoff_str = "no cutoff"

        ax.text(0.01, 0.9, cutoff_str,
                transform=ax.transAxes, fontsize=16)

    return f, axs


def make_bias_noise_figure(bias_cp_dfs, entropy_cp_dfs,
                           noisy_cp_dfs, noise_levels,
                           noiseless_cp_df, analytical_cp_df,
                           bias_cp_df_labels=None,
                           entropy_cp_df_labels=None,
                           base_figure_scale=3, include_binomial_null=False):

    num_noisy_cp_dfs = len(noisy_cp_dfs)

    normal_axis_scale = 5
    small_axis_scale = 2
    gap_scale = 1

    figsize = (num_noisy_cp_dfs * base_figure_scale, 2 * base_figure_scale)

    gridsize = [normal_axis_scale * 2 + small_axis_scale + gap_scale,
                num_noisy_cp_dfs * normal_axis_scale + (num_noisy_cp_dfs - 1) * gap_scale]

    f, ax_grid = plt.subplots(*gridsize, figsize=figsize)

    bias_ax = plt.subplot2grid(gridsize, (0, 0),
                               colspan=gridsize[-1] - normal_axis_scale - gap_scale,
                               rowspan=normal_axis_scale)

    counts = [bias_cp_df.groupby("index_set").count().final_loss
              for bias_cp_df in bias_cp_dfs]
    marginal_counts = [utils.compute_marginal_counts(count)
                       for count in counts]

    panels.make_bias_panel(bias_cp_dfs, counts, marginal_counts,
                           labels=bias_cp_df_labels, ax=bias_ax,
                           include_binomial_null=include_binomial_null)

    entropy_ax = plt.subplot2grid(gridsize, (0, gridsize[-1] - normal_axis_scale),
                                  rowspan=normal_axis_scale, colspan=normal_axis_scale)

    entropies = [utils.compute_entropy(cp_df) for cp_df in entropy_cp_dfs]
    entropy_sds = [utils.bootstrap_entropy_sd(cp_df) for cp_df in entropy_cp_dfs]

    panels.make_entropy_panel(entropy_cp_dfs, entropies, entropy_sds,
                              entropy_cp_df_labels, ax=entropy_ax)

    noise_axs = []

    for ii, (noisy_cp_df, noise_level) in enumerate(zip(noisy_cp_dfs, noise_levels)):
        if not noise_axs == []:
            subplot2grid_kwargs = {"sharex": noise_axs[0],
                                   "sharey": noise_axs[0]}
        else:
            subplot2grid_kwargs = {}

        noise_ax = plt.subplot2grid(gridsize,
                                    (normal_axis_scale + gap_scale,
                                     (normal_axis_scale + gap_scale) * ii),
                                    rowspan=normal_axis_scale,
                                    colspan=normal_axis_scale,
                                    **subplot2grid_kwargs)

        include_x_label = False

        if ii == 0:
            include_y_label = True
        else:
            include_y_label = False

        if ii + 1 == num_noisy_cp_dfs:
            include_legend = True
        else:
            include_legend = False

        panels.make_noise_comparison_panel(noisy_cp_df, analytical_cp_df, noiseless_cp_df,
                                           ax=noise_ax, include_legend=include_legend,
                                           include_x_label=include_x_label,
                                           include_y_label=include_y_label,
                                           noise_level=noise_level)

        if subplot2grid_kwargs != {}:
            subplot2grid_kwargs = {"sharex": noise_axs[0]}
        histogram_ax = plt.subplot2grid(gridsize,
                                        (normal_axis_scale * 2 + gap_scale,
                                         (normal_axis_scale + gap_scale) * ii),
                                        rowspan=small_axis_scale,
                                        colspan=normal_axis_scale,
                                        **subplot2grid_kwargs)

        cps = [noisy_cp_df, analytical_cp_df]
        colors = [NUMERICAL_CPS_COLOR, ANALYTICAL_CPS_COLOR]

        panels.make_histogram_comparison_panel(cps, colors,
                                               ax=histogram_ax,
                                               include_x_label=True,
                                               include_y_label=include_y_label)

        noise_axs.append(noise_ax)

    return f, ax_grid, bias_ax, noise_axs
