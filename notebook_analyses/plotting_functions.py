import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

from aracna.src.analysis.plot_comparison import apply_montserrat
from aracna.src.src.task_info.cat_paired import SupervisedTrainInfo


def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r / 255, g / 255, b / 255)


@apply_montserrat
def plot_inputs(
    inp,
    pos_info=None,
    read_ylim=None,
    window_size=None,
    c=None,
    remove_some_labels=True,
):
    import matplotlib.pyplot as plt

    inp = inp.squeeze(0)

    titles = ["", ""]
    ylabels = ["read count", "BAF"]

    res = [*inp.T]

    n_cols = 2

    fig, axs = plt.subplots((len(titles) + 1) // n_cols, n_cols, figsize=(6, 2.5))

    j = 0

    if pos_info is not None:
        chrom_starts = pos_info.groupby("chr").apply(lambda x: x.index.min()).values
        chrom_ends = pos_info.groupby("chr").apply(lambda x: x.index.max()).values
        chromosomes = pos_info["chr"].unique()
        midpoints = [(start + end) / 2 for start, end in zip(chrom_starts, chrom_ends)]

    for i, ax in enumerate(axs.flatten()):
        color_arr = np.full((res[j].shape[0], 3), hex_to_rgba("#4B6188"))
        if c is not None:
            color_arr = np.where(
                c[:, None], hex_to_rgba("#4B6188"), hex_to_rgba("#C32200")
            )
        ax.scatter(range(res[j].shape[0]), res[j], s=2, c=color_arr)
        ax.set_xlim(0, inp.shape[0])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            if window_size is not None:
                # Calculate moving average
                read_moving_avg = np.convolve(
                    res[j], np.ones(window_size) / window_size, mode="valid"
                )
                half_window = window_size // 2
                ax.plot(
                    range(half_window, len(read_moving_avg) + half_window),
                    read_moving_avg,
                    color="#C32200",
                )

            if read_ylim is not None:
                ax.set_ylim(0, read_ylim)

        ax.set_xlabel("position")

        if pos_info is not None:
            ax.set_xticks(chrom_starts)
            ax.set_xticklabels([])
            ax.set_xlabel("chromosome position", labelpad=10)
            for midpoint, chrom in zip(midpoints, chromosomes):
                if chrom > 9 and chrom % 3 != 2 and remove_some_labels:
                    continue
                ax.text(
                    midpoint,
                    -0.05,
                    f"{chrom}",
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                )

        ax.set_title(titles[j])
        ax.set_ylabel(ylabels[j])
        j += 1
    fig.tight_layout()


@apply_montserrat
def plot_data(
    inp,
    targets,
    read_ylim=None,
    window_size=None,
    num_minor=None,
    num_major=None,
    figsize=(4, 6),
    ratio_factor=1 / 4,
    minor_adj=0.07,
    color_arr=None,
):
    import matplotlib.pyplot as plt

    inp = inp.squeeze(0)
    targets = targets.squeeze(0).cpu().long().permute(0, 1)[:, [1, 0]]
    titles = ["", ""] + [
        "minor parental",
        "major parental",
    ]
    ylabels = ["read count", "BAF", "minor CN", "major CN"]

    min_class, major_class = ((targets + 1).max(axis=0)).values.tolist()
    min_class = min_class if num_minor is None else num_minor + 1
    major_class = major_class if num_major is None else num_major + 1

    one_hot_num = max(min_class, major_class)
    one_hot = torch.nn.functional.one_hot(targets.T, num_classes=one_hot_num)
    one_hot = one_hot.permute(0, 2, 1)

    res = [
        *inp.T,
        *one_hot,
    ]

    fig, axs = plt.subplots(
        # (len(titles) + 1) // n_cols,
        4,
        1,
        figsize=figsize,
        # gridspec_kw={"height_ratios": [1,1, ratio_factor * (min_class - 1) + minor_adj, ratio_factor * (major_class - 1)]},
        gridspec_kw={
            "height_ratios": [
                1,
                1,
                ratio_factor * (min_class - 1) + minor_adj,
                ratio_factor * (major_class - 1),
            ]
        },
    )

    j = 0

    for i, ax in enumerate(axs.flatten()):
        if i > 1:
            num_classes = min_class if j == 2 else major_class

            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", ["white", "#08306b"], N=100
            )
            sns.heatmap(
                res[j][:num_classes],
                ax=ax,
                cmap=cmap,
                cbar=False,
            )
            ax.set_yticks([i + 0.5 for i in range(num_classes)])
            ax.set_yticklabels(
                [i for i in range(num_classes)],
                rotation=0,
            )
            ax.set_ylabel(ylabels[j], fontsize=10, labelpad=15)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)
        else:
            colors = sns.color_palette("Blues", n_colors=256)

            if color_arr is None:
                # "#4B6188"
                ax.scatter(range(res[j].shape[0]), res[j], s=0.5, color="#4B6188")
            else:
                ax.scatter(range(res[j].shape[0]), res[j], s=0.5, c=color_arr)

            ax.set_xlim(0, inp.shape[0])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0:
                if window_size is not None:
                    # Calculate moving average
                    read_moving_avg = np.convolve(
                        res[j], np.ones(window_size) / window_size, mode="valid"
                    )
                    half_window = window_size // 2
                    ax.plot(
                        range(half_window, len(read_moving_avg) + half_window),
                        read_moving_avg,
                        color="C32200",
                    )

                if read_ylim is not None:
                    ax.set_ylim(0, read_ylim)
            ax.set_ylabel(ylabels[j], fontsize=10)
        # ax.set_title(titles[j], fontsize=10)
        # ax.set_ylabel(ylabels[j], fontsize=10)
        if j == 3:
            ax.set_xlabel("position", fontsize=10)
            n_vals = res[j].shape[1]
            if n_vals >= 100000:  # Case for large range in the 100,000s
                xticks = np.arange(0, n_vals + 1, 200000)
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{int(tick)}" for tick in xticks], rotation=45)
            else:  # Case for smaller range in the 10,000s
                xticks = np.arange(0, n_vals + 1, 2000)
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{int(tick)}" for tick in xticks], rotation=0)
        else:
            ax.set_xticklabels([])
            ax.set_xticks([])
        j += 1
    fig.tight_layout(h_pad=0.1)


@apply_montserrat
def plot_outs(
    inp,
    out,
    targets,
    task: SupervisedTrainInfo,
    max_total=None,
    read_ylim=None,
    window_size=100,
    include_probs=False,
):
    import matplotlib.pyplot as plt

    if max_total is None:
        max_total = task.max_tot_cn

    inp = inp.squeeze(0)
    # input is tuple ew
    int_vals = out["copy_numbers"].squeeze(0).T
    # skip total CN
    out_probs = out["marginal_probs"][..., 1:, :].squeeze(0).permute(1, 2, 0)

    one_hot = torch.nn.functional.one_hot(int_vals, num_classes=task.max_tot_cn + 2)
    one_hot = one_hot.permute(0, 2, 1)

    if targets is not None:
        targets = targets.squeeze(0).cpu()

    targets_titles = ["True Major CN", "True Minor CN"]

    titles = (
        ["read depth", "BAF"]
        + (targets_titles if targets is not None else [])
        + [
            "Predicted Major CN,",
            "Predicted Minor CN",
        ]
    )

    prob_vals = [
        "Major Probability",
        "Minor Probability",
    ]
    
    titles += prob_vals if include_probs else []

    height_ratios = [1] + ([1] if targets is not None else []) + [1.5] + ([1.5] if include_probs else [])
    gridspec_kw = {"height_ratios": height_ratios}

    res = [
        *inp.T,
        *(targets.T if targets is not None else []),
        *one_hot,
        *(out_probs if include_probs else [])
    ]
    if targets is not None:
        y_max = targets.max(axis=0).values
        y_lim = max(targets.max(), min(10, int_vals.max())) + 0.2
        adjust_plots = 0
    else:
        y_max = int_vals.max(axis=0)
        y_lim = max(10, int_vals.max()) + 0.2
        adjust_plots = 2

    n_cols = 2
    fig, axs = plt.subplots(
        (len(titles) + 1) // n_cols, n_cols, figsize=(10, 10), gridspec_kw=gridspec_kw
    )
    j = 0
    for i, ax in enumerate(axs.flatten()):
        if i >= 2 and i < n_cols and i % n_cols == 0:
            ax.remove()
            continue

        if i > 3 - adjust_plots:
            max_val = min(task.surplus_cats[j % 2] + 1, max_total + 1)

            sns.heatmap(
                res[j][:max_val],
                ax=ax,
                cmap=LinearSegmentedColormap.from_list(
                    "custom_cmap", ["white", "#08306b"], N=100
                ),
                cbar_kws={"orientation": "horizontal", "pad": 0.1},
            )
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([i + 0.5 for i in range(max_val)])
            if j % 2 == 0:
                if max_total < task.max_tot_cn:
                    ax.set_yticklabels(
                        [f"{i}" for i in range(max_val)],
                        rotation=0,
                    )
                else:
                    ax.set_yticklabels(
                        [
                            f"{i}"
                            for i in range(max_val - 1)
                        ]
                        + [f"Tot CN > {task.max_tot_cn}"],
                        rotation=0,
                    )
            else:
                ax.set_yticklabels(
                    [i for i in range(max_val - 1)]
                    + (
                        [f"Tot CN > {task.max_tot_cn}"]
                        if max_total > task.max_tot_cn
                        else [max_val - 1]
                    ),
                    rotation=0,
                )

        else:
            ax.scatter(range(res[j].shape[0]), res[j], s=2, color="#4B6188")
            ax.set_xlim(0, inp.shape[0])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i > 1:
                ax.set_ylim(0, y_lim)
            elif i == 0:
                if window_size is not None:
                    # Calculate moving average
                    read_moving_avg = np.convolve(
                        res[j], np.ones(window_size) / window_size, mode="valid"
                    )
                    half_window = window_size // 2
                    ax.plot(
                        range(half_window, len(read_moving_avg) + half_window),
                        read_moving_avg,
                        color="#C32200",
                    )

                if read_ylim is not None:
                    ax.set_ylim(0, read_ylim)

        ax.set_title(titles[j])
        j += 1


@apply_montserrat
def plot_recon(
    recon_df,
    prefixes=["ascat", "battenberg", "aracna", "hmm_copy"],
    include_pos=True,
    read_ylim=None,
    palette=["#CC6677", "#882255", "#44AA99", "#332288"],
    remove_some_labels=True,
    window_size=500,
    bottom_val=1,
    titles=None,
):
    ylabels = ["read count", "BAF"]
    val_col_map = {"RD": "read_depth", "BAF": "BAF"}

    n_cols = 2
    n_rows = len(prefixes)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2 * n_rows))

    recon_df = recon_df.fillna(bottom_val)

    if include_pos:
        pos_info = recon_df[["chr", "position"]]
        chrom_starts = pos_info.groupby("chr").apply(lambda x: x.index.min()).values
        chrom_ends = pos_info.groupby("chr").apply(lambda x: x.index.max()).values
        chromosomes = pos_info["chr"].unique()
        midpoints = [(start + end) / 2 for start, end in zip(chrom_starts, chrom_ends)]

    flat_axs = axs.flatten()
    ax_counter = 0

    for i, model in enumerate(prefixes):
        for val, col in val_col_map.items():
            ax = flat_axs[ax_counter]
            ax_counter += 1
            if val == "BAF" and model == "hmm_copy":
                ax.remove()
                continue

            scatter_base = recon_df[col].values
            ax.scatter(
                range(scatter_base.shape[0]),
                scatter_base,
                s=2,
                color=hex_to_rgba("#4B6188"),
                alpha=0.2,
            )
            ax.set_xlim(0, scatter_base.shape[0])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if val == "RD":
                ax.set_yscale("symlog", linthresh=1)
                # ax.set_ylim(-0.1, 8e2)
                if window_size is not None:
                    # Calculate moving average
                    read_moving_avg = np.convolve(
                        scatter_base, np.ones(window_size) / window_size, mode="valid"
                    )
                    half_window = window_size // 2
                    ax.plot(
                        range(half_window, len(read_moving_avg) + half_window),
                        read_moving_avg,
                        alpha=0.5,
                        color="#000435",
                        zorder=2,
                    )
                ax.set_ylim(bottom_val, read_ylim)
                labels = [item.get_text() for item in ax.get_yticklabels()]
                new_labels = ["NA"] + labels[1:]
                ax.set_yticklabels(new_labels)

            recon = recon_df[f"{model}_{val}"].values
            ax.scatter(
                range(scatter_base.shape[0]), recon, s=1, color=palette[i], zorder=3
            )

            if include_pos:
                ax.set_xticks(chrom_starts)
                ax.set_xticklabels([])
                ax.set_xlabel("chromosome position", labelpad=10)
                for midpoint, chrom in zip(midpoints, chromosomes):
                    if (chrom > 12) and (chrom % 3 != 0) and (remove_some_labels):
                        continue
                    ax.text(
                        midpoint,
                        -0.05,
                        f"{chrom}",
                        ha="center",
                        va="top",
                        transform=ax.get_xaxis_transform(),
                        fontsize=8,
                    )
            else:
                ax.set_xlabel("position")
                ax.set_xticklabels([])

            if i != len(prefixes) - 1:
                ax.set_xlabel("")

            ax.set_ylabel(f"{val}")
            ax.set_title(f"{model if titles is None else titles[i]} recon")
    fig.tight_layout(h_pad=1.1)
