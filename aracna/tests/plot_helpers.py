import matplotlib.pyplot as plt
from aracna.analysis.tcga_data import aracna_from_case
from notebook_analyses.plotting_functions import plot_outs


def get_real_plot(
    inference_info,
    base_dir,
    case,
    sample_type="tumor",
    keep_loci_name=None,
    depth_type="rolling_avg_2000",
    max_len=10000,
    plot_len=10000,
    burn_in=0,
    **kwargs,
):
    res_dict = aracna_from_case(
        inference_info,
        base_dir,
        case,
        sample_type,
        keep_loci_name,
        depth_type,
        max_len,
        burn_in=burn_in,
    )
    task = inference_info.trained_model.task_info

    plot_outs(
        res_dict["input"][..., :plot_len, :],
        {
            "copy_numbers": res_dict["output"]["copy_numbers"][..., :plot_len, :],
            "marginal_probs": res_dict["output"]["marginal_probs"][
                ..., :plot_len, :, :
            ],
        },
        None,
        task,
        max_major=4,
        **kwargs,
    )
    est_string = ", ".join(
        [f"{k}: {v.item():.2f}" for k, v in res_dict["globals"].items()]
    )
    plt.suptitle(f"Estimated {est_string}")
    plt.tight_layout()
