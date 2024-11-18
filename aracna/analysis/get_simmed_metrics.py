import json
import pickle

import numpy as np
import torch

from aracna.analysis.comparison_other_tools import (
    get_baf_rmse,
    get_break_points,
    get_concordance,
    get_ploidy,
    get_recon,
    get_rmse,
    recon_mae,
)
from aracna.analysis.utils import get_datamodule_infer_info


# Custom serialization function
def custom_serializer(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Convert NumPy types to native Python types
    if isinstance(obj, (np.ndarray, list, tuple)):  # Handle arrays or lists
        return [custom_serializer(i) for i in obj]
    if isinstance(obj, dict):  # Handle dictionaries
        return {k: custom_serializer(v) for k, v in obj.items()}
    error_message = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(error_message)


def get_sim_result(inference_info, data=None):
    dm = inference_info.data_module
    dm.sampler.profile_sampler.sample_seqlen = False
    task = inference_info.data_module.task_info

    if data is None:
        data = next(iter(dm.val_dataloader()))

    pos, inp, targets, dynam, global_info = data

    out_dict = inference_info.get_prediction(
        pos, inp, dynam["sample_len"], processed=True
    )
    targets = task.process_targets(targets.to(task.device), dynam["sample_len"])

    seq_targs = targets

    seq_targs = torch.where(
        seq_targs.sum(-1, keepdim=True) <= task.max_tot_cn,
        seq_targs,
        task.surplus_cats,
    )

    out_dict |= {"targets": seq_targs}

    return out_dict, global_info, data


def get_metrics(out_dict, global_info):
    cns = out_dict["output"]["copy_numbers"].squeeze(0).cpu().numpy()
    targets = out_dict["targets"].squeeze(0).cpu().numpy()
    inputs = out_dict["input"].squeeze(0).cpu().numpy()

    real_purity = global_info["purity"].item()
    real_read_depth_per_cn = global_info["read_depth"].item()

    predicted_purity = out_dict["globals"]["purity"].item()
    predicted_read_per_cn = out_dict["globals"]["read_depth"].item()

    break_points = get_break_points(cns)
    baf_rmse = get_baf_rmse(inputs[:, 1], predicted_purity, cns)
    recon_rd = get_recon(predicted_read_per_cn, predicted_purity, cns)
    rd_mae = recon_mae(inputs[:, 0], recon_rd)

    ploidy = {
        "real": get_ploidy(real_purity, targets, tumor_only=True),
        "predicted": get_ploidy(predicted_purity, cns, tumor_only=True),
    }
    purity = {"real": real_purity, "predicted": predicted_purity}
    rd_per_cn = {"real": real_read_depth_per_cn, "predicted": predicted_read_per_cn}

    return {
        "concordance": {
            "major": get_concordance(cns[:, 0:1], targets[:, 0:1]),
            "minor": get_concordance(cns[:, 1:2], targets[:, 1:2]),
            "both": get_concordance(cns, targets),
            "total": get_concordance(
                cns.sum(axis=-1)[:, None], targets.sum(axis=-1)[:, None]
            ),
        },
        "rmse": {
            "major": get_rmse(cns[:, 0:1], targets[:, 0:1]),
            "minor": get_rmse(cns[:, 1:2], targets[:, 1:2]),
            "both": get_rmse(cns, targets),
            "total": get_rmse(cns.sum(axis=-1)[:, None], targets.sum(axis=-1)[:, None]),
        },
        "break_points": break_points,
        "baf_rmse": baf_rmse,
        "rd_mae": rd_mae,
        "ploidy": ploidy,
        "purity": purity,
        "rd_per_cn": rd_per_cn,
        "seq_len": cns.shape[0],
    }


def write_simmed_output(base_dir, model_key, seqlen=None, num_samples=100):
    infer_info = get_datamodule_infer_info(model_key, seqlen=seqlen)

    dict_ls = []
    data_ls = []
    for _ in range(num_samples):
        out_dict, global_info, data = get_sim_result(infer_info)
        dict_ls.append(get_metrics(out_dict, global_info))
        data_ls.append(data)

    out_file = f"{base_dir}/{model_key}_summary_output.json"
    with open(out_file, "w") as json_file:
        json.dump(dict_ls, json_file, default=custom_serializer)

    data_out_file = f"{base_dir}/simmed_data.pkl"
    with open(data_out_file, "wb") as file:
        pickle.dump(data_ls, file)


def write_simmed_output_from_file(base_dir, model_key, seqlen=None):
    infer_info = get_datamodule_infer_info(model_key, seqlen=seqlen)

    data_out_file = f"{base_dir}/simmed_data.pkl"
    with open(data_out_file, "rb") as file:
        data_ls = pickle.load(file)

    dict_ls = []
    for data in data_ls:
        out_dict, global_info, _ = get_sim_result(infer_info, data)
        dict_ls.append(get_metrics(out_dict, global_info))

    out_file = f"{base_dir}/{model_key}_summary_output.json"
    with open(out_file, "w") as json_file:
        json.dump(dict_ls, json_file, default=custom_serializer)
