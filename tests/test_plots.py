from aracna.analysis.tcga_data import get_sample
from aracna.analysis.utils import get_datamodule_infer_info, get_infer_info
from matplotlib import pyplot as plt
from notebook_analyses.plotting_utils import (
    get_plot_test_profile,
    get_read_depth_plot,
)

from tests.plot_helpers import get_real_plot


def get_test_profiles():
    # test_profiles
    return [
        [[1], [0]],
        [[1], [1]],
        [[2], [0]],
        [[1, 2, 1], [1, 1, 1]],
        [[1, 2, 1, 2], [1, 1, 1, 0]],
        [[1, 2, 1, 2, 1], [1, 1, 1, 2, 1]],
        [[1, 3, 1, 2], [1, 0, 1, 2]],
    ]


def test_normal_plot():
    # project-1/araCNA-models/60n8iuqx
    infer_info = get_datamodule_infer_info("yth1w1ou")
    get_read_depth_plot(
        15.0,
        inference_info=infer_info,
        purity=1,
        plot_n=10000,
        read_ylim=150,
        window_size=None,
    )
    plt.show()


def test_profile_plot():
    test_profiles = get_test_profiles()
    infer_info = get_datamodule_infer_info("yth1w1ou")
    get_plot_test_profile(
        test_profiles[0],
        read_depth=15.0,
        inference_info=infer_info,
        read_ylim=150,
        window_size=None,
    )
    plt.show()
    return


def test_real_plot():
    base_dir = "~/data/tcga_analysis/output"
    case = "8cad4217-5699-4735-9be3-fc0015a8d262"
    read_file, baf_file = get_sample(base_dir, case, "tumor")
    infer_info = get_infer_info("yth1w1ou")
    get_real_plot(infer_info, read_file, baf_file, read_ylim=100)
    plt.show()


if __name__ == "__main__":
    test_real_plot()
    test_profile_plot()
    test_normal_plot()
