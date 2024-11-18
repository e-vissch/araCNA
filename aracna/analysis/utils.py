import os

import numpy as np
import pandas as pd
import torch
from aracna.src.infer import InferenceInfo, get_checkpointed_model, get_trainer
from aracna.src.train import get_data

from aracna.analysis.break_point_optimisation import opt_num_break_pts


def get_data_wrapped(tm):
    config = tm.hparams
    config.pop("wandb")
    config.wandb = None
    config.trainer.global_batch_size = 1
    return get_data(tm.task_info, config.task, config.loader)


def get_datamodule_infer_info(code_name, infer=False, seqlen=int(6.5e5)):
    tm = get_checkpointed_model(
        f"aracna/araCNA-models/{code_name}/checkpoints/last.ckpt", infer=infer, task=
        "pretrained"
    )
    if seqlen is not None:
        tm.hparams.task.dataset._start_seqlen = seqlen

    tm.hparams.task.dataset.pop("max_major", "None")
    tm.hparams.task.dataset.max_total = tm.hparams.task.dataset.get("max_total", 6)

    dm = get_data_wrapped(tm=tm)
    dm.set_curr_batch_size(1)
    trainer = get_trainer()
    return InferenceInfo(code_name, tm, trainer, dm)


def get_infer_info(code_name, infer=True, task=None):
    tm = get_checkpointed_model(
        f"aracna/araCNA-models/{code_name}/checkpoints/last.ckpt", infer=infer, task=task
    )
    tm.hparams.task.dataset._start_seqlen = 10000
    trainer = get_trainer()
    return InferenceInfo(code_name, tm, trainer, None)


def get_result(inference_info, pos, inp, max_len=10000, int_dtype=torch.int32):
    max_len = min(
        max_len - 1,
        inference_info.trained_model.task_info.max_seq_length - 1,
        pos.shape[1],
    )

    pos, inp = pos[:, :max_len], inp[:, :max_len]

    # for globals inputs
    # MUST concat w same type otherwise strange behaviour occurs
    pos = torch.concatenate([pos, torch.zeros(1, 1, 2, dtype=int_dtype)], dim=1)
    inp = torch.concatenate([inp, torch.zeros(1, 1, 2)], dim=1)

    return inference_info.get_prediction(
        pos, inp, torch.tensor([max_len]), processed=False
    )


def get_smoothed_cats(cat_probs, window_size=500):
    tensor = cat_probs.squeeze()

    if window_size % 2 == 0:
        window_size += 1

    # Ensure the tensor is 2D
    half_window = window_size // 2

    # Calculate the cumulative sum with padding at the beginning
    cumsum = torch.cat([torch.zeros(1, tensor.size(1)), tensor.cumsum(dim=0)], dim=0)

    # Compute the rolling mean for the full window size
    rolling_mean_full = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Handle the start and end of the tensor separately
    rolling_mean_start = torch.zeros(half_window, tensor.size(1))
    rolling_mean_end = torch.zeros(half_window, tensor.size(1))

    for i in range(1, half_window + 1):
        rolling_mean_start[i - 1] = cumsum[i * 2 - 1] / (i * 2)
        rolling_mean_end[-i] = (cumsum[-1] - cumsum[-(i * 2 + 1)]) / (i * 2)

    # Concatenate all parts
    rolling_mean = torch.cat(
        [rolling_mean_start, rolling_mean_full, rolling_mean_end], dim=0
    )

    return rolling_mean.argmax(dim=-1)


def get_marginal_info(marginal_probs, infer_info):
    exceed_cat = []
    max_maj = infer_info.max_trained_tot_CN
    max_min = infer_info.max_trained_tot_CN // 2

    if infer_info.max_trained_tot_CN > infer_info.can_train_upto_tot_CN:
        exceed_cat = [f"prob_tot_CN>{infer_info.can_train_upto_tot_CN}"]
        max_maj = infer_info.can_train_upto_tot_CN
        max_min = infer_info.can_train_upto_tot_CN // 2

    cnames_tot = [
        f"marg_prob_tot_{i}"
        for i in range(min(max_maj + max_min, infer_info.can_train_upto_tot_CN) + 1)
    ] + exceed_cat

    cnames_maj = [f"marg_prob_maj_{i}" for i in range(max_maj + 1)] + exceed_cat
    cnames_min = [f"marg_prob_min_{i}" for i in range(max_min + 1)] + exceed_cat

    return {
        c_name: marginal_probs[0, :, 0, t] for t, c_name in enumerate(cnames_tot)
    } | {
        c_name: marginal_probs[0, :, j + 1, i]  # 1 as first dim is total
        for j, ls in enumerate([cnames_maj, cnames_min])
        for i, c_name in enumerate(ls)
    }


def get_categorical_info(cat_probs, infer_info):
    def get_c_name(maj, min):
        if maj <= infer_info.max_trained_tot_CN:
            if maj == infer_info.task_info.max_tot_cn + 1:
                return f"Tot>{infer_info.task_info.max_tot_cn}"
            return f"prob_maj_min_{maj}_{min}"

    return {
        cname: cat_probs[0, :, cat]
        for (cat, (maj, min)) in infer_info.task_info.target_category_dict.items()
        if (cname := get_c_name(maj, min)) is not None
    }



def get_detailed_df(infer_info, base_dict, res_dict, window_sizes):
    aracna_df = pd.DataFrame(
        base_dict | 
        {
            "raw_major_CN": res_dict["output"]["copy_numbers"][0, :, 0],
            "raw_minor_CN": res_dict["output"]["copy_numbers"][0, :, 1],
        }
        | get_categorical_info(res_dict["output"]["orig_cat_probs"], infer_info)
        | get_marginal_info(res_dict["output"]["marginal_probs"], infer_info)
    )
    for w in window_sizes:
        new_seq, _ = opt_num_break_pts(
            res_dict["output"]["orig_cat_probs"].squeeze().numpy(), w
        )
        aracna_df[
            [f"major_smoothed_window_opt_{w}", f"minor_smoothed_window_opt_{w}"]
        ] = new_seq

    return aracna_df


def get_aracna_dfs(infer_info, res_dict, default_window=500, detailed=False, window_sizes=(250, 500)):
    base_dict = {
            "chr": res_dict["positional_info"][0, :, 1],
            "position": res_dict["positional_info"][0, :, 0],
            "read_depth": res_dict["input"][0, :, 0],
            "BAF": res_dict["input"][0, :, 1]
            }

    smoothed_seq, _ = opt_num_break_pts(
        res_dict["output"]["orig_cat_probs"].squeeze().numpy(), default_window
    )
    base_dict["major_CN"], base_dict["minor_CN"] = smoothed_seq[:, 0], smoothed_seq[:, 1]


    if detailed:
        aracna_df = get_detailed_df(infer_info, base_dict, res_dict, window_sizes)
    else:
        aracna_df = pd.DataFrame(base_dict)

    aracna_df[["major_CN", "minor_CN"]] = aracna_df[["major_CN", "minor_CN"]].astype(
        float
    )
    aracna_df.loc[
        aracna_df["major_CN"] == infer_info.can_train_upto_tot_CN + 1,
        ["major_CN", "minor_CN"],
    ] = np.nan
    aracna_df.loc[
        aracna_df["major_CN"] == infer_info.can_train_upto_tot_CN + 1, "comment"
    ] = f"tot CN est > {infer_info.can_train_upto_tot_CN}"

    return aracna_df, pd.DataFrame(res_dict["globals"])


def write_aracna_csvs(infer_info, res_dict, out_stub, detailed=False):
    aracna_df, globals_df = get_aracna_dfs(infer_info, res_dict, detailed=detailed)
    os.makedirs(os.path.dirname(out_stub), exist_ok=True)
    aracna_df.to_csv(
        f"{out_stub}aracna_results_{infer_info.code_name}.csv", index=False
    )
    globals_df.to_csv(
        f"{out_stub}aracna_globals_{infer_info.code_name}.csv", index=False
    )
    return aracna_df, globals_df
