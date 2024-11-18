from .cat_paired import PairedInfo, SupervisedTrainInfo, UnsupervisedTrainInfo
from .task_info import TaskInfo

registry = {
    "simple": TaskInfo,
    "paired": SupervisedTrainInfo,
    "unsupervised_paired": UnsupervisedTrainInfo,
    "infer_paired": PairedInfo,
}
