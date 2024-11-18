import functools
import warnings
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

# Path to the Montserrat font file
font_path = f"{os.path.dirname(__file__)}/Montserrat-Regular.ttf"  # Update this path

# Add the font to Matplotlib's font manager
fm.fontManager.addfont(font_path)


def apply_montserrat(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        plt.rcParams["font.family"] = "Montserrat"
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10

        # Call the original function
        return func(*args, **kwargs)

    return wrapper


@apply_montserrat
def get_plot_from_val_list(
    joined_df,
    val_list,
    titles,
    model_keys,
    read_ylim=300,
    save_file=None,
    window_size=100,
    include_prob=True,
    colors=None,
    max_vals=None,
    prob_scale=1.5,
    include_chrom=True,
    max_total=8,
    subsample=None,
):
    row_adjust = int(len(val_list[0]) > 1)

    if subsample is not None:
        joined_df = joined_df.head(subsample)

    n_other_rows = len(val_list) + row_adjust
    n_rows = n_other_rows + int(include_prob)
    n_cols = 3
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(3 * n_cols, 2 * n_rows),
        gridspec_kw={
            "height_ratios": [1] * n_other_rows + [prob_scale] * int(include_prob)
        },
    )
    res = [joined_df["read_depth"].values, joined_df["BAF"].values]

    _max_vals = []
    max_adjust = []
    for vals in val_list:
        rel_df = joined_df[vals]
        na_mask = torch.zeros(rel_df.shape[0], dtype=torch.bool)
        if rel_df.isnull().any().any():
            warnings.warn(
                f"Some of {vals} have missing/not-a-number values, will include NA category"
            )
            na_mask = rel_df.isna().any(axis=1)
            max_adjust.append(1)
            includes_na = True
        else:
            max_adjust.append(0)
            includes_na = False

        int_vals = torch.tensor(rel_df.fillna(0).values).long()
        _max_vals.append(int_vals.max().item())
        if len(vals) == 1:  # hmm
            one_hot_tot = torch.nn.functional.one_hot(int_vals.squeeze()).T
            res += [one_hot_tot]
            if includes_na:
                new_dim = torch.zeros(1, one_hot_tot.shape[-1])
                one_hot_tot = torch.cat([new_dim, one_hot_tot], dim=0)
                one_hot_tot[0, na_mask] = 1
                one_hot_tot[1, na_mask] = 0
            continue

        one_hot = torch.nn.functional.one_hot(int_vals.T).permute(0, 2, 1)
        tot = int_vals.sum(axis=-1)
        # tot[na_mask] = -1
        one_hot_tot = torch.nn.functional.one_hot(tot).T

        if includes_na:
            new_dim = torch.zeros(2, 1, one_hot.shape[-1])
            one_hot = torch.cat([new_dim, one_hot], dim=1)
            one_hot[:, 0, na_mask] = 1
            one_hot[:, 1, na_mask] = 0  # as nas have been set to 0
            one_hot_tot = torch.cat([new_dim[0], one_hot_tot], dim=0)
            one_hot_tot[0, na_mask] = 1
            one_hot_tot[1, na_mask] = 0

        res += [*one_hot, one_hot_tot]

    if include_prob:
        for model_key in model_keys:
            key_stub = f"{model_key}_" if len(model_keys) > 1 else ""
            # TODO if statements for backward compat, delete on new trained
            aracna_maj_vals = joined_df[
                [
                    c
                    for i in range(max_total)
                    if (c := f"{key_stub}marg_prob_maj_{i}") in joined_df.columns
                ]
            ].values
            aracna_min_vals = joined_df[
                [
                    c
                    for i in range(max_total // 2)
                    if (c := f"{key_stub}marg_prob_min_{i}") in joined_df.columns
                ]
            ].values
            aracna_tot_vals = joined_df[
                [
                    c
                    for i in range(max_total)
                    if (c := f"{key_stub}marg_prob_tot_{i}") in joined_df.columns
                ]
            ].values
            _max_vals.append(5)

            max_adjust.append(0)
            res += [aracna_maj_vals.T, aracna_min_vals.T, aracna_tot_vals.T]

    if include_chrom:
        chrom_starts = joined_df.groupby("chr").apply(lambda x: x.index.min()).values
        chrom_ends = joined_df.groupby("chr").apply(lambda x: x.index.max()).values
        chromosomes = joined_df["chr"].unique()
        midpoints = [(start + end) / 2 for start, end in zip(chrom_starts, chrom_ends)]

    i = -1
    for j, ax in enumerate(axes.flatten()):
        if j == 2 and len(val_list[0]) > 1:
            ax.remove()
            continue
        i += 1
        ax.set_title(titles[i])
        if j <= 1:
            ax.scatter(range(res[j].shape[0]), res[i], s=2, color="#4B6188")
            ax.set_xlim(0, res[j].shape[0])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # for k, (chrom, start, end) in enumerate(zip(chromosomes, chrom_starts, chrom_ends)):
            #     ax.axvline(x=start, color='blue', linestyle='--')    # Mark the end of a

            if j == 0:
                if window_size is not None:
                    # Calculate moving average
                    read_moving_avg = np.convolve(
                        res[i], np.ones(window_size) / window_size, mode="valid"
                    )
                    half_window = window_size // 2
                    ax.plot(
                        range(half_window, len(read_moving_avg) + half_window),
                        read_moving_avg,
                        color="#C32200",
                    )
                ax.set_ylim(0, read_ylim)
        else:
            # if (j + 1) % 3 == 0 or j // 3 < 3:
            idx = j // 3 - row_adjust
            if (j + 1) % 3 == 0:
                cutoff_val = 9
            elif max_vals:
                cutoff_val = max(5, min(max_vals[idx] + 1, _max_vals[idx] + 1))
            else:
                cutoff_val = max(5, min(10, _max_vals[idx] + 1))
            cbar = False
            cbar_kws = None
            if j // 3 == len(val_list) + row_adjust:
                cbar = True
                cbar_kws = {"orientation": "horizontal", "pad": 0.15}

            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", ["white", "#08306b"], N=100
            )
            if colors is not None:
                c_val = colors[j // 3]
                if c_val is not None:
                    cmap = LinearSegmentedColormap.from_list(
                        "custom_cmap", ["white", c_val], N=100
                    )

            num_classes = cutoff_val + max_adjust[idx]
            sns.heatmap(
                res[i][:num_classes],
                ax=ax,
                cmap=cmap,
                cbar=cbar,
                cbar_kws=cbar_kws,
            )
            ax.set_yticks([i + 0.5 for i in range(num_classes)])
            labels = [i for i in range(cutoff_val)]
            if max_adjust[idx] == 1:
                labels = ["NA"] + labels
            ax.set_yticklabels(
                labels,
                rotation=0,
            )

        ax.set_xticklabels([])
        ax.set_xticks([])

        if j // 3 == len(val_list) + row_adjust - 1:
            if include_chrom:
                ax.set_xlabel("chromosome position", labelpad=10)
            else:
                ax.set_xlabel("position", labelpad=10)

        if include_chrom:
            # Manually set the labels at the midpoints
            ax.set_xticks(chrom_starts)
            for midpoint, chrom in zip(midpoints, chromosomes):
                if (chrom > 12) and (chrom % 3 != 0):
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

    fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=300)


@apply_montserrat
def get_plot_from_joined(
    joined_df,
    model_keys,
    read_ylim=300,
    windows=(500),
    save_file=None,
    window_size=100,
    model_mappings={},
):
    joined_df["chr"] = joined_df["chr"].astype(int)
    joined_df = joined_df.sort_values(["chr", "position"])

    val_list = [
        ["hmm_state"],
        ["ascat_nMajor", "ascat_nMinor"],
        ["bat_nMaj1_A", "bat_nMin1_A"],
        ["cnvkit_cn1", "cnvkit_cn2"],
        *[
            [
                f"{model_key}_major_smoothed_window_opt_{w}",
                f"{model_key}_minor_smoothed_window_opt_{w}",
            ]
            for model_key in model_keys
            for w in windows
        ],
    ]

    def get_model_str(model_key):
        extra_info = model_mappings.get(model_key)
        return f"{extra_info}" if extra_info is not None else model_key

    def get_window_titles(model_key, w):
        w_round = 5 * round(w / 5)
        stub = f"araCNA-{get_model_str(model_key)}\nsmooth {w_round}"
        return [f"major {stub}", f"minor {stub}", f"total {stub}"]

    def get_probs(model_key):
        stub = f"araCNA-{get_model_str(model_key)} probs"
        return [f"major {stub}", f"minor {stub}", f"total {stub}"]

    titles = (
        ["read depth", "BAF", "hmm copy"]
        + ["major ascat", "minor ascat", "total ascat"]
        + ["major battenberg", "minor battenberg", "total battenberg"]
        + ["major CNV-kit", "minor CNV-kit", "total CNV-kit"]
        + [
            t
            for model_key in model_keys
            for w in windows
            for t in get_window_titles(model_key, w)
        ]
        + [t for model_key in model_keys for t in get_probs(model_key)]
    )

    get_plot_from_val_list(
        joined_df, val_list, titles, model_keys, read_ylim, save_file, window_size
    )
