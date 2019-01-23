import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt


N = 16
K = 4

ANALYTICAL_CPS_COLOR = "gray"
NUMERICAL_CPS_COLOR = "chartreuse"
LABEL_FONTSIZE = 18


def make_trajectory_panel(cp_df, failed_cp_df, ax=None,
                          include_x_label=True, include_y_label=True, include_legend=False,
                          title=None):

    if ax is None:
        f, ax = plt.figure(figsize=(6, 6))

    plot_trajectories(cp_df, "squared_grad_norms", ax=ax,
                      plot_func_kwargs={"color": "k", "alpha": 0.45})

    if failed_cp_df is not None:
        plot_trajectories(failed_cp_df, "squared_grad_norms", ax=ax,
                          plot_func_kwargs={"color": "C1", "alpha": 0.45})

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([1e-40, 1e1])
    ax.set_xlim([1, 50000])

    if include_x_label:
        ax.set_xlabel(r"Epoch", fontsize=LABEL_FONTSIZE)
    if include_y_label:
        ax.set_ylabel(r"$\vert\vert \nabla L(\theta) \vert\vert^2 $", fontsize=LABEL_FONTSIZE)
    if include_legend:
        legend_lines = [matplotlib.lines.Line2D([0], [0], color="k", lw=2),
                        matplotlib.lines.Line2D([0], [0], color="C1", lw=2)]

        ax.legend(legend_lines, ['successful runs', 'failed runs'], loc="lower left")

    if title is not None:
        ax.set_title(title, fontsize=24)

    return ax


def make_loss_index_panel(cp_df, analytical_cp_df, ax=None,
                          color_by=None, cmap="Set3",
                          include_x_label=True,
                          include_y_label=True,
                          include_legend=False):

    if ax is None:
        f, ax = plt.figure(figsize=(6, 6))

    ax.scatter(analytical_cp_df.morse_index, analytical_cp_df.cost,
               color=ANALYTICAL_CPS_COLOR, alpha=0.65, label="analytical")

    if color_by is not None:
        ax.scatter(cp_df.morse_index, cp_df.candidate_loss,
                   c=cp_df[color_by], cmap=cmap, label="numerical")
    else:
        ax.scatter(cp_df.morse_index, cp_df.candidate_loss,
                   label="numerical",
                   color=NUMERICAL_CPS_COLOR)

    if include_x_label:
        ax.set_xlabel("Index", fontsize=LABEL_FONTSIZE)
    if include_y_label:
        ax.set_ylabel(r"$L(\theta)$", fontsize=LABEL_FONTSIZE)

    if include_legend:
        ax.legend(loc="lower right")

    return ax


def make_performance_panel_pair(cp_df, analytical_cp_df,
                                failed_cp_df=None,
                                axs=None, title=None,
                                color_by=None, cmap="Set3",
                                include_x_label=True,
                                include_y_label=True,
                                include_legend=False,
                                plot_func_kwargs=None):
    if axs is None:
        f, axs = plt.subplots(ncols=2, figsize=(6, 12))

    traj_ax, loss_index_ax = axs

    make_trajectory_panel(cp_df, failed_cp_df, ax=traj_ax,
                          include_x_label=include_x_label,
                          include_y_label=include_y_label,
                          title=title,
                          include_legend=include_legend)

    make_loss_index_panel(cp_df, analytical_cp_df, ax=loss_index_ax,
                          color_by=color_by, cmap=cmap,
                          include_x_label=include_x_label,
                          include_y_label=include_y_label,
                          include_legend=include_legend)

    return axs


def make_histogram_comparison_panel(cp_dfs, colors,
                                    ax=None, include_x_label=True,
                                    include_y_label=True):

    if ax is None:
        f, ax = plt.subplots(ncols=1, figsize=(6, 6))

    for cp_df, color in zip(cp_dfs, colors):
        ax.hist(cp_df.morse_index, density=True,
                histtype="stepfilled", linewidth=3, alpha=0.1, color=color)
        ax.hist(cp_df.morse_index, density=True,
                histtype="step", linewidth=3, alpha=1, color=color)

    if include_x_label:
        ax.set_xlabel("Index", fontsize=LABEL_FONTSIZE)

    if include_y_label:
        ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)

    ax.set_yscale("log")


def make_noise_comparison_panel(cp_df, analytical_cp_df,
                                highlight_cp_df=None,
                                ax=None, title=None, include_legend=False,
                                include_x_label=False, include_y_label=False,
                                noise_level=None):
    if ax is None:
        f, ax = plt.subplots(ncols=1, figsize=(6, 6))

    ax.scatter(
        analytical_cp_df.morse_index, analytical_cp_df.cost,
        label="analytical", color=ANALYTICAL_CPS_COLOR, alpha=0.5)

    if highlight_cp_df is not None:
        ax.scatter(
            highlight_cp_df.morse_index, highlight_cp_df.candidate_loss,
            label="numerical, no noise", color='k',
            marker='o', facecolor="none", s=72, linewidth=2)

    ax.scatter(
        cp_df.morse_index, cp_df.candidate_loss,
        label="numerical, noise",
        color=NUMERICAL_CPS_COLOR)

    if include_x_label:
        ax.set_xlabel("Index", fontsize=LABEL_FONTSIZE)
    if include_y_label:
        ax.set_ylabel(r"$L(\theta)$", fontsize=LABEL_FONTSIZE)

    if include_legend:
        plt.legend(loc='lower right')

    if noise_level is not None:
        ax.text(0.01, 0.9, r"$\sigma={0}$".format(noise_level),
                transform=ax.transAxes, fontsize=LABEL_FONTSIZE - 2)

    return ax


def make_bias_panel(bias_cp_dfs, counts, marginal_counts,
                    labels=None, include_binomial_null=False, ax=None):

    num_bias_cp_dfs = len(bias_cp_dfs)

    if ax is None:
        f, ax = plt.subplots(figsize=(8, 4))

    if labels is None:
        labels = [""] * num_bias_cp_dfs

    colors = ["C" + str(ii) for ii in range(num_bias_cp_dfs)]

    for ii, (bias_cp_df, count, marginal_count, label, color) in enumerate(
        zip(
            bias_cp_dfs, counts, marginal_counts, labels, colors)):

        bar_nudge = (ii - num_bias_cp_dfs // 2) * (1 / (num_bias_cp_dfs + 1))
        bar_positions = np.arange(2, N + 2) + bar_nudge
        ax.bar(bar_positions, marginal_count, label=label, width=1 / (num_bias_cp_dfs + 1))
        ax.bar(bar_nudge, count.iloc[0], width=1 / (num_bias_cp_dfs + 1),
               color=color, edgecolor="k")

    xticks = [0] + list(range(2, N + 2))
    xticklabels = ["CP @ 0"] + list(range(N, 0, -1))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(r"Observed Count", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel(r"Eigenvector ID", fontsize=LABEL_FONTSIZE)

    if include_binomial_null:
        ax.hlines(np.mean(marginal_count),
                  *[1, N], linewidth=4, label="Null Expected Height")
        sd = np.sqrt(np.mean(marginal_count) *
                     ((N - 1) / N))  # binomial standard deviation
        ax.hlines([np.mean(marginal_count) + sd,
                   np.mean(marginal_count) - sd], *[1, 16],
                  color="gray", linestyle="--", linewidth=2,
                  label=r"Null +/- SD")

    plt.legend(loc="upper left")


def make_entropy_panel(cp_dfs, entropies, entropy_sds, labels,
                       ax=None):

    if ax is None:
        f, ax = plt.subplots(figsize=(4, 4))

    num_bars = len(cp_dfs)

    for ii, (entropy, entropy_sd, label) in enumerate(
        zip(
            entropies, entropy_sds, labels)):

        ax.bar(ii, entropy, width=0.6, label=label, color="gray")
        ax.errorbar(ii, entropy, yerr=entropy_sd, color="k", elinewidth=2)

    ax.set_xticks(range(num_bars))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Entropy", fontsize=18)
    ax.set_ylim([2, np.log2(N + 1)])


def plot_trajectories(cp_df, key, plot_func="plot", func=lambda x: x, ax=None,
                      subplots_kwargs=None, plot_func_kwargs=None):

    if ax is None:
        if subplots_kwargs is None:
            subplots_kwargs = {}

        f, ax = plt.subplots(**subplots_kwargs)

    if plot_func_kwargs is None:
        plot_func_kwargs = {}

    hs = []
    
    for ii, row in cp_df.iterrows():
        ys = func(row[key])
        if plot_func == "plot":
            hs = ax.plot(ys, **plot_func_kwargs)

    return ax, hs
