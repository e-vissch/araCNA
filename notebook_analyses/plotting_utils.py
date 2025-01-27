import matplotlib.pyplot as plt
import pandas as pd
import torch
from aracna.analysis.comparison_other_tools import get_reconstruction_metrics, get_val_dict
from aracna.src.datamodules.simulated.main_cna_sampling_func import sample_from_profile

from notebook_analyses.plotting_functions import plot_outs


def inference(model, data, task):
    pos, input, targets, dynam, _ = data
    res_dict = model.inference((pos, input), dynam["sample_len"])

    targets = task.process_targets(targets, dynam["sample_len"])

    return res_dict | {"targets": targets}


def get_result_from_data(data, inference_info):
    task = inference_info.data_module.task_info
    pos, inp, targets, dynam, _ = data

    out_dict = inference_info.get_prediction(
        pos, inp, dynam["sample_len"], processed=True
    )
    all_targets = task.process_targets(targets.to(task.device), dynam["sample_len"])

    seq_targets = all_targets[..., :2]

    seq_targets = torch.where(
        seq_targets.sum(-1, keepdim=True) <= task.max_tot_cn,
        seq_targets,
        task.surplus_cats,
    )

    out_dict |= {"targets": seq_targets}

    return (
        *out_dict.values(),
        task,
    )


def plot_from_data(
    _pos_info, input_dat, output, pred_globs, targets, task, true_globs, plot_n, **kwargs
):
    plot_outs(
        input_dat[..., :plot_n, :],
        {
            k: res[..., :plot_n, :]
            if k != "marginal_probs"
            else res[..., :plot_n, :, :]
            for k, res in output.items()
        },
        targets[..., :plot_n, :],
        task,
        **kwargs,
    )
    est_string = ", ".join([f"{k}: {v.item():.2f}" for k, v in pred_globs.items()])
    true_string = ", ".join([f"{k}: {v:.2f}" for k, v in true_globs.items()])

    plt.suptitle(f"Estimated {est_string}, True {true_string}")
    plt.tight_layout()


def get_outs_for_read_params(read_depth, inference_info, purity=1):
    dm = inference_info.data_module
    dm.sampler.profile_sampler.read_depth_range = [read_depth, read_depth]
    dm.sampler.profile_sampler.purity_range = [purity, purity]
    dm.sampler.profile_sampler.sample_seqlen = False
    data = next(iter(dm.val_dataloader()))

    data[-1] |= {
        "read_depth": torch.tensor([read_depth]),
        "purity": torch.tensor([purity]),
    }
    return get_result_from_data(data, inference_info)


def get_read_depth_plot(read_depth, inference_info, purity=1, plot_n=1000, **kwargs):
    plot_from_data(
        *get_outs_for_read_params(read_depth, inference_info, purity=purity),
        true_globs={"read_depth": read_depth, "purity": purity},
        plot_n=plot_n,
        max_total=inference_info.data_module.sampler.profile_sampler.max_total,
        **kwargs,
    )


def get_model_inference(profile, seqlen, read_depth, inference_info, purity=1):
    dm = inference_info.data_module
    rd_scale_range = (
        inference_info.trained_model.hparams.task.dataset.read_depth_scale_range
    )
    baf_scale_range = inference_info.trained_model.hparams.task.dataset.baf_scale_range
    res = sample_from_profile(
        profile, seqlen, seqlen + 1, read_depth, purity, rd_scale_range, baf_scale_range
    )

    read_depth_loc = min(
        seqlen, dm.task_info.max_seq_length - 1
    )  # with minus 1 indexing.

    data = dm.task_info.process_inputs(
        *res,
        {"sample_len": torch.tensor([read_depth_loc])},
        {
            "read_depth": torch.tensor([read_depth]),
            "purity": torch.tensor([purity]),
        },
    )

    return get_result_from_data(data, inference_info)


def get_plot_test_profile(
    profile, read_depth, inference_info, purity=1, seq_len=10000, plot_n=10000, **kwargs
):
    plot_from_data(
        *get_model_inference(
            profile, seq_len, read_depth, inference_info, purity=purity
        ),
        true_globs={"read_depth": read_depth, "purity": purity},
        plot_n=plot_n,
        max_total=inference_info.data_module.sampler.profile_sampler.max_total,
        **kwargs,
    )


def get_recon_df(joined_df, global_df, aracna_remap):
    joined_df = joined_df.reset_index()
    val_dict = get_val_dict(joined_df)
    recon_dict = get_reconstruction_metrics(
        val_dict,
        joined_df.read_depth,
        joined_df.BAF,
        joined_df,
        global_df,
        mean=False,
        return_recon=True,
    )
    tot_df = pd.concat({k: pd.DataFrame(v) for k, v in recon_dict.items()}, axis=0)
    tot_df = tot_df.reset_index(level=0).rename(columns={"level_0": "metric_name"})

    tot_df = tot_df[
        [
            c
            for c in tot_df.columns
            if ("aracna" not in c or c in list(aracna_remap.keys()))
        ]
    ]
    tot_df = tot_df.rename(columns=aracna_remap)
    tot_df_pivot = tot_df.pivot_table(index=tot_df.index, columns="metric_name")

    # Flatten the columns and add the prefix from 'metric_name'
    tot_df_pivot.columns = [f"{metric}_{col}" for metric, col in tot_df_pivot.columns]

    keep_cols = ["chr", "position", "read_depth", "BAF"]
    for c in keep_cols:
        tot_df_pivot[c] = joined_df[c]
    return tot_df_pivot
