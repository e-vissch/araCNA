import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from patsy import dmatrix

from aracna.analysis.plot_comparison import get_plot_from_val_list
from aracna.analysis.utils import get_infer_info, get_result, write_aracna_csvs

MASTER_CHR_VAL = "chr"


def set_chrom_colnam(df):
    df.columns = [c.lower() for c in df.columns]
    POSS_CHROM_VALS = ["chr", "chromosome", "chrom"]
    df.rename(columns={c: MASTER_CHR_VAL for c in POSS_CHROM_VALS}, inplace=True)


def get_relevant_df(read_fname, baf_fname):
    depth_df = pd.read_csv(read_fname)
    set_chrom_colnam(depth_df)
    depth_df[MASTER_CHR_VAL] = depth_df[MASTER_CHR_VAL].astype(str)

    depth_df[MASTER_CHR_VAL] = (
        depth_df[MASTER_CHR_VAL]
        .str.replace("X|Y", lambda x: "23" if x.group() == "X" else "24", regex=True)
        .astype(int)
    )

    baf_df = pd.read_csv(baf_fname, delimiter="\t")
    set_chrom_colnam(baf_df)

    baf_df = baf_df.rename(columns={baf_df.columns[-1]: "BAF"})
    baf_df[MASTER_CHR_VAL] = (
        baf_df[MASTER_CHR_VAL].apply(
            lambda x: "23" if x == "X" else "24" if x == "Y" else x
        )
    ).astype(int)

    merge_cols = [MASTER_CHR_VAL, "position"]

    return (
        pd.merge(depth_df, baf_df[[*merge_cols + "BAF"]], on=merge_cols)
        .dropna()
        .reset_index()
    )


def get_gc_corrected_rd(data_df, depth_cname):
    cols_upto1kb = [name for name in data_df.columns if re.search(r"\d+bp$|1kb$", name)]
    cols_over5kb = [
        name
        for name in data_df.columns
        if re.search(r"(?<!\d)(?:[5-9]\d*|[1-9]\d{2,})kb$|\d+mb$", name)
    ]

    def get_corr(col, gc_df):
        corrs = [np.abs(gc_df[gc_col].corr(col)) for gc_col in gc_df.columns]
        idx = np.argmax(corrs)
        # can't begin w no below
        return {f"col_{gc_df.columns[idx]}": gc_df[gc_df.columns[idx]]}

    corr_low = get_corr(data_df[depth_cname], data_df[cols_upto1kb])
    corr_high = get_corr(data_df[depth_cname], data_df[cols_over5kb])

    fit_data = pd.DataFrame(corr_low | corr_high | {"y": data_df[depth_cname]})
    # Generate natural splines
    spline_basis = dmatrix(
        f"bs({next(iter(corr_low))}, df=5, include_intercept=True) + bs({next(iter(corr_high))}, df=5, include_intercept=True)",
        fit_data,
        return_type="dataframe",
    )
    # Fit the linear model
    model = sm.OLS(fit_data.y, spline_basis).fit()
    return np.maximum(model.resid + model.params[0], 0)


def get_torched_tcga_data(
    input_dat_name,
    depth_cname="depth",
    float_dtypes=torch.float32,
    int_dtypes=torch.int32,
):
    input_df = pd.read_csv(input_dat_name)
    input_df = input_df.dropna()

    pos_feats, inputs = (
        (input_df["position"], input_df[MASTER_CHR_VAL]),
        (input_df[depth_cname], input_df["BAF"]),
    )
    return (
        torch.tensor(np.stack(pos_feats, axis=-1), dtype=int_dtypes).unsqueeze(0),
        torch.tensor(np.stack(inputs, axis=-1), dtype=float_dtypes).unsqueeze(0),
    )


def get_sample(base_dir, case, sample_type="tumor"):
    aracna_dir = f"{base_dir}/{case}/aracna"
    return f"{aracna_dir}/{sample_type}_BAF_rd.txt"


def plot_aracna_single(aracna_df, model_key, plot_file, include_prob=False):
    val_list = [
        ["major_smoothed_window_opt_500", "minor_smoothed_window_opt_500"],
    ]

    titles = (
        ["read depth", "BAF"]
        + [
            f"major araCNA-{model_key}",
            f"minor araCNA-{model_key}",
            f"total araCNA-{model_key}",
        ]
        + [
            f"major probs\naraCNA-{model_key}",
            f"minor probs\naraCNA-{model_key}",
            f"total probs\naraCNA-{model_key}",
        ]
    )

    get_plot_from_val_list(
        aracna_df,
        val_list,
        titles,
        [model_key],
        max_vals=[8, 8, 8],
        prob_scale=1.2,
        save_file=plot_file,
        include_prob=include_prob,
    )


def aracna_from_case(
    infer_info,
    base_dir,
    case,
    sample_type="tumor",
    depth_type="depth",
    max_len=int(1e6),
):
    input_file = get_sample(base_dir, case, sample_type)
    pos, inp = get_torched_tcga_data(input_file, depth_cname=depth_type)
    return get_result(infer_info, pos, inp, max_len)


def write_case_csvs(
    out_stub,
    model_key,
    input_file,
    depth_type="depth",
    max_len=int(1e6),
    task="pretrained",
    detailed=False,
    include_plot=False,
):
    infer_info = get_infer_info(model_key, task=task)
    pos, inp = get_torched_tcga_data(input_file, depth_cname=depth_type)
    result = get_result(infer_info, pos, inp, max_len)
    aracna_df, globals_df = write_aracna_csvs(infer_info, result, out_stub=out_stub, detailed=detailed)
    if include_plot:
        plot_aracna_single(aracna_df, model_key, f"{out_stub}_plot.png")


def write_cases_csvs_from_lists(
    out_stubs,
    model_key,
    input_files,
    depth_type="depth",
    max_len=int(1e6),
    task="pretrained",
    detailed=False,
    include_plot=False,
):
    infer_info = get_infer_info(model_key, task=task)

    for input_file, out_stub in zip(input_files, out_stubs):
        pos, inp = get_torched_tcga_data(input_file, depth_cname=depth_type)
        result = get_result(infer_info, pos, inp, max_len)
        aracna_df, globals_df = write_aracna_csvs(infer_info, result, out_stub=out_stub, detailed=detailed)
        if include_plot:
            plot_aracna_single(aracna_df, model_key, f"{out_stub}_plot.png")
