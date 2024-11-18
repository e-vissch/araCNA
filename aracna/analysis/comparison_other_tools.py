import glob
import json
import os
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

from aracna.analysis.break_point_optimisation import get_break_points
from aracna.analysis.plot_comparison import get_plot_from_joined

TOTAL_ONLY_KEYS = ["hmm_copy", "cnv_tot", "cnv_tot_purity_ascat"]


def join_together(df_to_join, other_df, join_cols, prefix, join=""):
    # note, df_to_join and other_df referenced in query string, so they are used.
    query_string = f"""
    SELECT df_to_join.*, {','.join(
        f'segments.{col} as {prefix}_{col}' for col in join_cols
        )}
    FROM other_df segments
    {join}JOIN df_to_join df_to_join
    ON segments.chr = df_to_join.chr
    AND df_to_join.position BETWEEN segments.startpos AND segments.endpos
    """

    return duckdb.query(query_string).to_df()


def get_joined_df_using_query(
    ascat_df, battenberg_df, hmm_df, cnv_df, cnv_pa_df, aracna_df
):
    joined_df = join_together(
        aracna_df, ascat_df, ["nMajor", "nMinor"], "ascat", "RIGHT "
    )
    joined_df = join_together(
        joined_df,
        battenberg_df,
        [
            "nMaj",
            "nMin",
            "nMaj1_A",
            "nMin1_A",
            "frac1_A",
            "nMaj2_A",
            "nMin2_A",
            "frac2_A",
        ],
        "bat",
        "RIGHT ",
    )
    joined_df = join_together(joined_df, hmm_df, ["state"], "hmm", "RIGHT ")
    joined_df = join_together(
        joined_df, cnv_df, ["cn", "cn1", "cn2"], "cnvkit", "RIGHT "
    )
    joined_df = join_together(
        joined_df, cnv_pa_df, ["cn", "cn1", "cn2"], "cnvkit_pa", "RIGHT "
    )
    return joined_df.sort_values(by=["chr", "position"])


def get_other_df(fname, chr_val="chr"):
    model_df = pd.read_csv(fname, sep="\t")
    model_df = model_df.rename(columns={"start": "startpos", "end": "endpos"})
    model_df = model_df[
        model_df[chr_val].isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])
    ]
    model_df["chr"] = (
        (model_df[chr_val].str.replace("chr", "")).replace({"X": 23}).astype(int)
    )
    return model_df


def get_aracna_dfs(aracna_dir, aracna_prefix, aracna_keys):
    for i, aracna_key in enumerate(aracna_keys):
        aracna_f = f"{aracna_dir}/{aracna_prefix}results_{aracna_key}.csv"
        aracna_df = (
            pd.read_csv(aracna_f)
            .set_index(["chr", "position", "read_depth", "BAF"])
            .add_prefix(f"{aracna_key}_")
            .reset_index(level=["read_depth", "BAF"])
        )
        if i == 0:
            result_df = aracna_df
        else:
            aracna_df.drop(columns=["read_depth", "BAF"], inplace=True)
            result_df = result_df.join(aracna_df)

    return result_df.reset_index()


def get_dfs(
    ascat_dir, battenberg_dir, hmm_dir, cnv_dir, aracna_dir, aracna_keys, aracna_prefix
):
    ascat_f = glob.glob(os.path.join(ascat_dir, "*.segments.txt"))[0]
    battenberg_f = glob.glob(os.path.join(battenberg_dir, "*subclones.txt"))[0]

    ascat_df = pd.read_csv(ascat_f, sep="\t")
    ascat_df["chr"] = ascat_df["chr"].replace("X", 23).astype(int)

    battenberg_df = pd.read_csv(battenberg_f, sep="\t")
    battenberg_df["chr"] = (
        (battenberg_df["chr"].str.replace("chr", "")).replace("X", 23).astype(int)
    )
    battenberg_df[["nMaj", "nMin"]] = np.where(
        battenberg_df[["frac1_A"]] >= 0.5,
        battenberg_df[["nMaj1_A", "nMin1_A"]],
        battenberg_df[["nMaj2_A", "nMin2_A"]],
    )

    cnv_df = get_other_df(f"{cnv_dir}/tumor_pure.cns.baf_call", "chromosome")
    cnv_pa_df = get_other_df(f"{cnv_dir}/tumor_ascat_purity.cns.baf_call", "chromosome")

    hmm_df = get_other_df(f"{hmm_dir}/hmm_copy.tsv")

    aracna_dfs = get_aracna_dfs(aracna_dir, aracna_prefix, aracna_keys)

    return ascat_df, battenberg_df, hmm_df, cnv_df, cnv_pa_df, aracna_dfs


def get_glob_info(ascat_dir, battenberg_dir, aracna_dir, aracna_keys, aracna_prefix):
    ascat_globals = pd.read_csv(f"{ascat_dir}/qc.csv")
    battenberg_globals = pd.read_csv(
        glob.glob(os.path.join(battenberg_dir, "*_rho_and_psi.txt"))[0], sep="\t"
    )
    aracna_globals_ls = [
        pd.read_csv(f"{aracna_dir}/{aracna_prefix}globals_{aracna_key}.csv")
        for aracna_key in aracna_keys
    ]

    joined_globals = pd.DataFrame(
        {
            "purity": [
                ascat_globals.purity.item(),
                battenberg_globals.loc["FRAC_GENOME", "rho"],
                *[aracna_globals.purity.item() for aracna_globals in aracna_globals_ls],
            ],
            "ploidy": [
                ascat_globals.ploidy.item(),
                battenberg_globals.loc["FRAC_GENOME", "ploidy"],
                *[None for _ in aracna_globals_ls],
            ],
            "read_depth_per_cn": [
                None,
                None,
                *[
                    aracna_globals.read_depth.item()
                    for aracna_globals in aracna_globals_ls
                ],
            ],
        },
        index=["ascat", "battenberg"]
        + [f"aracna_{model_key}" for model_key in aracna_keys],
    )

    return joined_globals


def get_sample_cns(purity, parental_cn):
    return (1 - purity) + purity * parental_cn


def get_robust_mean(rd, trim_ratio, up_factor=3):
    lower_val = np.quantile(rd, trim_ratio)
    upper_val = np.quantile(rd, 1 - trim_ratio * up_factor)
    mean_mask = (rd >= lower_val) & (rd <= upper_val)
    return rd[mean_mask].mean(), mean_mask


def get_ploidy(purity, parental_cn, include_locs=None, tumor_only=False):
    if tumor_only:
        base_tot = parental_cn.sum(axis=-1)
    else:
        base_tot = get_sample_cns(purity, parental_cn).sum(axis=-1)
    mask = ~np.isnan(parental_cn).any(axis=1)
    if include_locs is not None:
        mask = (include_locs) & (mask)
    return base_tot[mask].mean()


def get_recon(avg_rd_per_cn, purity, parental_cn):
    tot_cn = get_sample_cns(purity, parental_cn).sum(axis=-1)
    return avg_rd_per_cn * tot_cn


def get_approx_recon(
    read_depth, purity, ploidy, parental_cn, trim_ratio=0.05, use_ploidy=True
):
    if use_ploidy:
        avg_rd, mask = get_robust_mean(read_depth, trim_ratio)
        avg_rd_per_cn = avg_rd / (ploidy * purity + 2 * (1 - purity))
    else:
        avg_rd, mask = get_robust_mean(read_depth, trim_ratio)
        avg_rd_per_cn = avg_rd / get_ploidy(purity, parental_cn, mask)
    return get_recon(avg_rd_per_cn, purity, parental_cn)


def get_bat_sample_cns(purity, bat_vals):
    tumor_cns = (
        bat_vals[["bat_nMaj1_A", "bat_nMin1_A"]].values
        * bat_vals[["bat_frac1_A"]].values
        + bat_vals[["bat_nMaj2_A", "bat_nMin2_A"]].values
        * bat_vals[["bat_frac2_A"]].values
    )
    return tumor_cns * (purity) + (1 - purity)


def get_bat_recon(read_depth, purity, bat_vals, trim_ratio=0.05):
    tot_samp_cn = get_bat_sample_cns(purity, bat_vals).sum(axis=1)
    avg_rd, mask = get_robust_mean(read_depth, trim_ratio)
    avg_rd_per_cn = avg_rd / np.nanmean(tot_samp_cn[mask])
    return avg_rd_per_cn * tot_samp_cn


def get_approx_tot_recon(read_depth, total_cn, purity=1, trim_ratio=0.05):
    samp_cn = get_sample_cns(purity, total_cn)
    avg_rd, mask = get_robust_mean(read_depth, trim_ratio)
    avg_rd_per_cn = avg_rd / samp_cn[mask].mean()
    return get_recon(avg_rd_per_cn, purity, samp_cn[:, None])


def get_baf_rmse(
    baf_values, purity, parental_cn, bat_df=None, mean=True, return_recon=False
):
    haplotypes = np.array([[0.0, 0], [0, 1], [1, 0], [1, 1]])
    if bat_df is not None:
        sample_cns = get_bat_sample_cns(purity, bat_df)
    else:
        if not return_recon:
            mask = ~np.isnan(parental_cn).any(axis=1)
            parental_cn = parental_cn[mask]
            baf_values = baf_values[mask]
        sample_cns = get_sample_cns(purity, parental_cn)
    total_out = sample_cns.sum(axis=1)
    mask = total_out != 0
    poss_minor_c = (sample_cns[None, :, :] * haplotypes[:, None, :]).sum(axis=-1)
    poss_vals = np.zeros_like(poss_minor_c)
    poss_vals[:, mask] = poss_minor_c[:, mask] / total_out[mask]
    recon_diff = poss_vals - baf_values[None, :]
    arg_val = (recon_diff**2).argmin(axis=0)

    if return_recon:
        return poss_vals[arg_val, np.arange(recon_diff.shape[1])]

    recon_loss = recon_diff[arg_val, np.arange(recon_diff.shape[1])]
    if mean:
        return (recon_loss**2).mean() ** 0.5
    return recon_loss


def recon_mae(read_depth, read_depth_recon, mean=True):
    if mean:
        return (np.abs(read_depth_recon - read_depth)).mean()
    return read_depth_recon - read_depth


def get_reconstruction_metrics(
    val_dict,
    read_depth,
    baf_values,
    joined_df,
    global_info,
    mean=True,
    return_recon=False,
):

    extra_cs = ["bat_nMaj2_A", "bat_nMin2_A", "bat_frac2_A"]
    bat_vals = joined_df[["bat_nMaj1_A", "bat_nMin1_A", "bat_frac1_A"] + extra_cs]
    bat_vals[extra_cs] = bat_vals[extra_cs].fillna(0)

    def _get_purity(key):
        if "aracna" in key:
            key = "_".join(key.split("_")[:2])  # get with model key
        elif key == "cnv_both":
            return 1
        elif key == "cnv_both_purity_ascat":
            return global_info.loc["ascat", "purity"]
        return global_info.loc[key, "purity"]

    baf_results = (
        {
            key: get_baf_rmse(
                baf_values.values,
                _get_purity(key),
                val,
                mean=mean,
                bat_df=(None if key != "battenberg" else bat_vals),
                return_recon=return_recon,
            )
            for key, val in val_dict.items()
            if key not in TOTAL_ONLY_KEYS
        }
        | {
            "battenberg_approx": get_baf_rmse(
                baf_values.values,
                global_info.loc["battenberg", "purity"],
                joined_df[["bat_nMaj", "bat_nMin"]].values,
                mean=mean,
                return_recon=return_recon,
            )
        }
        | {
            "cnv_both_purity_ascat": get_baf_rmse(
                baf_values.values,
                global_info.loc["ascat", "purity"],
                val_dict["cnv_both_purity_ascat"],
                mean=mean,
                bat_df=None,
                return_recon=return_recon,
            )
        }
    )

    rd_recon = {
        "ascat": get_approx_recon(
            read_depth,
            global_info.loc["ascat", "purity"],
            global_info.loc["ascat", "ploidy"],
            val_dict["ascat"],
        ),
        "battenberg_approx": get_approx_recon(
            read_depth,
            global_info.loc["battenberg", "purity"],
            global_info.loc["battenberg", "ploidy"],
            joined_df[["bat_nMaj", "bat_nMin"]].values,
        ),
        "battenberg": get_bat_recon(
            read_depth,
            global_info.loc["battenberg", "purity"],
            bat_vals,
        ),
        "cnv_both": get_approx_recon(
            read_depth,
            1,
            np.nanmean(val_dict["cnv_tot"]),
            val_dict["cnv_both"],
        ),
        "cnv_both_purity_ascat": get_approx_recon(
            read_depth,
            global_info.loc["ascat", "purity"],
            np.nanmean(val_dict["cnv_tot_purity_ascat"]),
            val_dict["cnv_both_purity_ascat"],
        ),
        "hmm_copy": get_approx_tot_recon(read_depth, val_dict["hmm_copy"]),
        "cnv_tot": get_approx_tot_recon(read_depth, val_dict["cnv_tot"]),
        "hmm_copy_purity_ascat": get_approx_tot_recon(
            read_depth, val_dict["hmm_copy"], global_info.loc["ascat", "purity"]
        ),
        "cnv_tot_purity_ascat": get_approx_tot_recon(
            read_depth,
            val_dict["cnv_tot_purity_ascat"],
            global_info.loc["ascat", "purity"],
        ),
    } | {
        key: get_recon(
            global_info.loc["_".join(key.split("_")[:2]), "read_depth_per_cn"],
            global_info.loc["_".join(key.split("_")[:2]), "purity"],
            val_dict[key],
        )
        for key in val_dict
        if "aracna" in key
    }

    if return_recon:
        return {"BAF": baf_results, "RD": rd_recon}

    rd_mae = {
        key: recon_mae(read_depth, val, mean=mean) for key, val in rd_recon.items()
    }

    return {"BAF_RMSE": baf_results, "RD_MAE": rd_mae}


def get_concordance(val1, val2):
    mask = ~np.isnan(val1).all(axis=-1) & ~np.isnan(val2).all(axis=-1)
    return (val1[mask] == val2[mask]).all(axis=-1).mean()


def get_rmse(val1, val2):
    mask = ~np.isnan(val1) & ~np.isnan(val2)
    return ((val1[mask] - val2[mask]) ** 2).mean() ** 0.5


def get_concordance_and_mse(val_dict, suffix=""):
    concordance_dict = defaultdict(dict)
    mse_dict = defaultdict(dict)

    for key1, val1 in val_dict.items():
        for key2, val2 in val_dict.items():
            if key1 == key2:
                continue
            if key1 in TOTAL_ONLY_KEYS or key2 in TOTAL_ONLY_KEYS:
                val_tot_1 = val1.sum(axis=-1) if key1 not in TOTAL_ONLY_KEYS else val1
                val_tot_2 = val2.sum(axis=-1) if key2 not in TOTAL_ONLY_KEYS else val2
                concordance_dict[key1][key2] = {
                    "total": get_concordance(val_tot_1[:, None], val_tot_2[:, None]),
                }
                mse_dict[key1][key2] = {
                    "total": get_rmse(val_tot_1, val_tot_2),
                }
                continue

            concordance_dict[key1][key2] = {
                "both": get_concordance(val1, val2),
                "total": get_concordance(
                    val1.sum(axis=1)[:, None], val2.sum(axis=1)[:, None]
                ),
                "major": get_concordance(val1[:, 0:1], val2[:, 0:1]),
                "minor": get_concordance(val1[:, 1:], val2[:, 1:]),
            }
            mse_dict[key1][key2] = {
                "both": get_rmse(val1, val2),
                "total": get_rmse(val1.sum(axis=1)[:, None], val2.sum(axis=1)[:, None]),
                "major": get_rmse(val1[:, 0:1], val2[:, 0:1]),
                "minor": get_rmse(val1[:, 1:], val2[:, 1:]),
            }

    return {f"concordance{suffix}": concordance_dict, f"rmse{suffix}": mse_dict}


def get_break_points_dict(val_dict):
    return {
        "num_break_points": {
            val: get_break_points(arr, val not in TOTAL_ONLY_KEYS)
            for val, arr in val_dict.items()
        }
    }


def get_dirs(root_dir, ascat_correction="TRUE", penalty=70):
    ascat_dir = f"{root_dir}/ascat_wgs/corr_{ascat_correction}penalty_{penalty}"
    battenberg_dir = f"{root_dir}/battenberg"
    hmm_dir = f"{root_dir}/hmm_copy"
    cnv_dir = f"{root_dir}/cnv_kit"
    aracna_dir = f"{root_dir}/aracna"
    return ascat_dir, battenberg_dir, hmm_dir, cnv_dir, aracna_dir


def get_val_dict(joined_df):
    ascat_vals = joined_df[["ascat_nMajor", "ascat_nMinor"]].values
    battenberg_vals = joined_df[["bat_nMaj", "bat_nMin"]].values

    hmm_vals = joined_df["hmm_state"].values

    cnv_vals = joined_df[["cnvkit_cn1", "cnvkit_cn2"]].values
    cnv_vals_pa = joined_df[["cnvkit_pa_cn1", "cnvkit_pa_cn2"]].values

    vals = [
        c
        for c in joined_df.columns
        if ("major_smoothed_window" in c) or "major_CN" in c
    ]

    return {
        "ascat": ascat_vals,
        "battenberg": battenberg_vals,
        "hmm_copy": hmm_vals,
        "cnv_both": cnv_vals,
        "cnv_tot": joined_df["cnvkit_cn"].values,
        "cnv_both_purity_ascat": cnv_vals_pa,
        "cnv_tot_purity_ascat": joined_df["cnvkit_pa_cn"].values,
    } | {
        f'aracna_{val.replace("major_smoothed_window_", "win_").replace("major_CN", "")}': joined_df[
            [val, val.replace("major", "minor")]
        ].values
        for val in vals
    }


def get_normalised_val_dict(val_dict, rounded=False):
    def get_norm(vals, sum_dim=True):
        apply_func = round if rounded else lambda x: x
        if sum_dim:
            ploidy = apply_func(vals[~np.isnan(vals).any(axis=1)].sum(axis=1).mean())
        else:
            ploidy = apply_func(vals[~np.isnan(vals)].mean())

        return (vals - ploidy) / ploidy

    return {
        key: get_norm(vals, key not in TOTAL_ONLY_KEYS)
        for key, vals in val_dict.items()
    }


def write_combined(
    root_dir,
    aracna_keys,
    aracna_prefix,
    out_prefix="",
    ascat_correction="TRUE",
    drop_x=True,
):
    ascat_dir, battenberg_dir, hmm_dir, cnv_dir, aracna_dir = get_dirs(
        root_dir, ascat_correction
    )

    ascat_df, battenberg_df, hmm_df, cnv_df, cnv_pa_df, aracna_dfs = get_dfs(
        ascat_dir,
        battenberg_dir,
        hmm_dir,
        cnv_dir,
        aracna_dir,
        aracna_keys,
        aracna_prefix,
    )
    joined_df = get_joined_df_using_query(
        ascat_df, battenberg_df, hmm_df, cnv_df, cnv_pa_df, aracna_dfs
    )

    if drop_x:
        joined_df = joined_df[joined_df.chr != 23].reset_index(drop=True)

    joined_dir = f"{root_dir}/joined_results"
    os.makedirs(joined_dir, exist_ok=True)

    aracna_str = "_".join(sorted(aracna_keys))

    joined_df.to_csv(f"{joined_dir}/{out_prefix}sequence_{aracna_str}.csv", index=False)

    joined_globals = get_glob_info(
        ascat_dir, battenberg_dir, aracna_dir, aracna_keys, aracna_prefix
    )
    joined_globals.to_csv(f"{joined_dir}/{out_prefix}globals_{aracna_str}.csv")
    return joined_df, joined_globals


def get_purity_ploidy(joined_globals, val_dict, aracna_keys):
    aracna_ids = [f"aracna_{key}" for key in aracna_keys]

    rel_keys = {
        c: aracna_id for c in val_dict for aracna_id in aracna_ids if aracna_id in c
    }

    purity = {
        "ascat": joined_globals.loc["ascat", "purity"],
        "battenberg": joined_globals.loc["battenberg", "purity"],
    } | {a_id: joined_globals.loc[a_id, "purity"] for a_id in aracna_ids}

    tumor_ploidy = (
        {
            c: get_ploidy(joined_globals.loc[c, "purity"], val_dict[c], tumor_only=True)
            for c in ["ascat", "battenberg"]
        }
        | {
            "hmm_copy": val_dict["hmm_copy"].mean(),
            "cnv_kit": np.nanmean(val_dict["cnv_tot"]),
            "cnv_kit_purity_ascat": get_ploidy(
                joined_globals.loc["ascat", "purity"],
                val_dict["cnv_both_purity_ascat"],
                tumor_only=True,
            ),
        }
        | {
            c: get_ploidy(
                joined_globals.loc[aracna_key, "purity"], val_dict[c], tumor_only=True
            )
            for c, aracna_key in rel_keys.items()
        }
    )

    total_ploidy = (
        {
            c: get_ploidy(joined_globals.loc[c, "purity"], val_dict[c])
            for c in ["ascat", "battenberg"]
        }
        | {
            "hmm_copy": val_dict["hmm_copy"].mean(),
            "cnv_kit": np.nanmean(val_dict["cnv_tot"]),
            "cnv_kit_purity_ascat": get_ploidy(
                joined_globals.loc["ascat", "purity"], val_dict["cnv_both_purity_ascat"]
            ),
        }
        | {
            c: get_ploidy(joined_globals.loc[aracna_key, "purity"], val_dict[c])
            for c, aracna_key in rel_keys.items()
        }
    )

    reported_ploidy = {
        "ascat": joined_globals.loc["ascat", "ploidy"],
        "battenberg": joined_globals.loc["battenberg", "ploidy"],
    }

    return {
        "purity": purity,
        "ploidy": tumor_ploidy,
        "reported_ploidy": reported_ploidy,
        "total_ploidy": total_ploidy,
    }


def get_segment_lengths(val_dict):
    return_dict = {}
    for key, _arr in val_dict.items():

        arr = _arr[:, None] if key in TOTAL_ONLY_KEYS else _arr

        # Identify the changes
        if key in TOTAL_ONLY_KEYS:
            arr = arr[:, None]

        change = np.any(np.diff(arr, axis=0) != 0, axis=1)

        # Prepend a False value to align the array lengths
        change = np.insert(change, 0, False)

        # Calculate the cumulative sum to create groups
        group = np.cumsum(change)

        # Count the number of rows in each group
        counts = np.bincount(group)
        return_dict[key] = counts

    return return_dict


def analyse_results(
    joined_df,
    joined_globals,
    model_keys,
    out_json_file=None,
    save_plot_file=None,
    include_plot=True,
):
    tot_val_dict = get_val_dict(joined_df)

    num_na_dict = {
        "num_na": {
            key: np.isnan(arr).sum() / len(arr.shape)
            for key, arr in tot_val_dict.items()
        }
    }

    # don't do intersect w cnv kit as too many
    intersect = joined_df.dropna(
        subset=["ascat_nMajor", "ascat_nMinor", "bat_nMaj", "bat_nMin", "hmm_state"]
        + [
            f"{model_key}_{cn_type}_CN"
            for model_key in model_keys
            for cn_type in ["major", "minor"]
        ]
    )

    val_dict = get_val_dict(intersect)

    recon_dict = get_reconstruction_metrics(
        val_dict, intersect.read_depth, intersect.BAF, intersect, joined_globals
    )
    concordance_dict = get_concordance_and_mse(val_dict)

    concordance_dict_norm = get_concordance_and_mse(
        get_normalised_val_dict(val_dict), "_norm"
    )
    concordance_dict_norm_r = get_concordance_and_mse(
        get_normalised_val_dict(val_dict, rounded=True), "_norm_rounded"
    )

    tot_dict = (
        recon_dict
        | concordance_dict
        | get_break_points_dict(val_dict)
        | get_purity_ploidy(joined_globals, val_dict, model_keys)
        | concordance_dict_norm
        | concordance_dict_norm_r
        | num_na_dict
    )
    # Save dictionary as JSON file

    if out_json_file:
        with open(out_json_file, "w") as json_file:
            json.dump(tot_dict, json_file)

    if include_plot:
        get_plot_from_joined(joined_df, model_keys, save_file=save_plot_file)

    return tot_dict


def get_aracna_str(aracna_keys):
    return "_".join(sorted(aracna_keys))


def write_analysis(
    root_dir, aracna_keys, joined_df, joined_globals, include_plot, out_prefix
):
    aracna_str = get_aracna_str(aracna_keys)
    out_json_file = (
        f"{root_dir}/joined_results/{out_prefix}summary_stats_{aracna_str}.json"
    )
    save_plot_file = (
        f"{root_dir}/joined_results/{out_prefix}{aracna_str}.png"
        if include_plot
        else None
    )
    analyse_results(
        joined_df,
        joined_globals,
        aracna_keys,
        out_json_file,
        save_plot_file,
        include_plot,
    )


def analyse_results_from_dir(root_dir, aracna_keys, include_plot, file_prefix):
    aracna_str = get_aracna_str(aracna_keys)
    joined_df = pd.read_csv(
        f"{root_dir}/joined_results/{file_prefix}sequence_{aracna_str}.csv"
    )
    joined_globals = pd.read_csv(
        f"{root_dir}/joined_results/{file_prefix}globals_{aracna_str}.csv", index_col=0
    )
    write_analysis(
        root_dir, aracna_keys, joined_df, joined_globals, include_plot, file_prefix
    )


def get_joined_analyse_results(
    root_dir,
    aracna_keys,
    aracna_prefix,
    include_plot=True,
    file_prefix="",
    ascat_correction="TRUE",
    drop_x=True,
):
    joined_df, joined_globals = write_combined(
        root_dir, aracna_keys, aracna_prefix, file_prefix, ascat_correction, drop_x
    )
    write_analysis(
        root_dir, aracna_keys, joined_df, joined_globals, include_plot, file_prefix
    )
