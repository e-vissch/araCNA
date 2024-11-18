from dataclasses import dataclass

from aracna.src.metrics.metrics import mse
from aracna.src.models.decoders import SimpleCnaDecoder


@dataclass
class TaskInfo:
    decoder_cls = SimpleCnaDecoder  # static class attribute, not input.
    targets_included = True  # static class attribute, not input.

    def __post_init__(self):
        # needed for MRO in inheritance, again- kinda gross?
        pass

    @staticmethod
    def remove_extra(rem_val, prepend_len=0, len_no_postpend=None):
        return rem_val[:, prepend_len:len_no_postpend]

    def metrics(self, metric_names, *args, prefix=None, **kwargs):
        print_prefix = "" if prefix is None else f"{prefix}/"
        metric_dict = {}
        for metric_name, metric_fn in metric_names.items():
            metric_result = metric_fn(*args, **kwargs, prefix=prefix)
            if isinstance(metric_result, dict):
                metric_dict.update({k: v.mean() for k, v in metric_result.items()})
            else:
                metric_dict[f"{print_prefix}{metric_name}"] = metric_result.mean()

        return metric_dict

    def process_inputs(self, *args):
        return args

    def process_for_analyses(self, input, output):
        return {"input": input, "output": output}


@dataclass
class TrainTaskInfo(TaskInfo):
    def default_loss(self, _input, output, targets, sample_len):
        return mse(output, targets, sample_len)

    def process_for_loss(self, batch, output):
        return self.process_for_analyses(*batch, output)


@dataclass
class SeqInfo(TaskInfo):
    max_seq_length: int = 1000
    prepend_len: int = 0
    postpend_len: int = 0

    @property
    def max_fillable_seq_length(self):
        return self.max_seq_length - self.prepend_len - self.postpend_len

    @property
    def len_no_postpend(self):
        return self.max_seq_length - self.postpend_len

    def process_for_analyses(self, input, output, **kwargs):
        input, output = (
            self.remove_extra(
                remove_val,
                prepend_len=self.prepend_len,
            )
            for remove_val in [input, output]
        )
        return {"input": input, "output": output} | kwargs


@dataclass
class TrainSeqInfo(SeqInfo, TrainTaskInfo):
    # MRO means SeqInfo -> TrainTaskInfo -> TaskInfo
    def process_for_loss(self, input, output, targets, **kwargs):
        base_dict = self.process_for_analyses(input, output, **kwargs)
        targets = self.remove_extra(
            targets,
            prepend_len=self.prepend_len,
        )

        return {"targets": targets} | base_dict
