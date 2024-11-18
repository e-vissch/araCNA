from aracna.analysis.get_simmed_metrics import get_sim_result
from aracna.analysis.utils import get_aracna_dfs, get_datamodule_infer_info
import matplotlib.pyplot as plt
from aracna.analysis.plot_comparison import get_plot_from_val_list

def get_res_from_targets(infer_info):
    result = get_sim_result(infer_info)
    aracna_df, globals_df = get_aracna_dfs(infer_info, result[0])
    targets = result[0]['targets'].squeeze()
    aracna_df["true_major"] = targets[:, 0].cpu()
    aracna_df["true_minor"] = targets[:, 1].cpu()
    return aracna_df, globals_df


def get_plot(total, model_key, plot_n=None):
    val_list = [["true_major", "true_minor"],
                [f'major_CN', f'minor_CN']]

    titles = (
        ["read depth", "BAF"]
        + ["true major CN", "true minor CN", "true total CN"]
        + [f"major araCNA-{model_key}", f"minor araCNA-{model_key}", f"total araCNA-{model_key}"]
    )
    
    plot_df = total if plot_n is None else total.head(plot_n)

    get_plot_from_val_list(plot_df, val_list, titles, [model_key], read_ylim=260, window_size=None, max_vals=[8, 8], prob_scale=1.2, include_chrom=False, include_prob=False)



def get_simulated_infer(model_key, read_depth, purity):
    infer_info = get_datamodule_infer_info(model_key)

    sampler = infer_info.data_module.sampler.profile_sampler
    sampler.read_depth_range = (read_depth, read_depth)
    sampler.purity_range = (purity, purity)

    sampler.read_depth_scale_range = (0.1, 0.1)
    sampler.baf_scale_range = (0.05, 0.05)

    aracna_df, global_df = get_res_from_targets(infer_info)
    
    get_plot(aracna_df, model_key)
    
    true_globs = {"read depth": read_depth, "purity": purity}
    pred_globs = {"read depth": global_df.read_depth, "purity": global_df.purity}
    est_string = ", ".join([f"{k}: {v.item():.2f}" for k, v in pred_globs.items()])
    true_string = ", ".join([f"{k}: {v:.2f}" for k, v in true_globs.items()])

    plt.suptitle(f"Estimated {est_string}, True {true_string}")
    plt.tight_layout()
    plt.show()
