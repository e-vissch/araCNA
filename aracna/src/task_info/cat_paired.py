from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import torch
import torch.nn.functional as F

from aracna.src.metrics.metrics import get_stacked_batch_loss
from aracna.src.models.decoders import PairedClassificationGlobalDecoder
from aracna.src.task_info.task_info import SeqInfo, TaskInfo, TrainSeqInfo, TrainTaskInfo
from aracna.src.utils.constants import (
    BAF_IN_DIM,
    GLOBAL_TOKEN,
    PURITY_KEY,
    READ_IN_DIM,
    READ_KEY,
    SEQ_TOKEN,
    TOKEN_IN_DIM,
)


@dataclass
class PairedInfo(TaskInfo):
    max_seq_length: int
    max_tot_cn_arch: int = 10

    max_tot_cn: int = 10  # must be <= max_tot_cn_arch

    prepend_len: int = 0
    loss_weights: Optional[list] = None

    base_info: SeqInfo = field(init=False)

    # avg_rd_trim_ratio: Optional[float] = None
    avg_rd_trim_ratio: Optional[float] = 0.05
    upper_trim_factor: Optional[int] = 1

    # class property
    decoder_cls = PairedClassificationGlobalDecoder

    @property
    def supervised_predict_keys(self):
        return [READ_KEY, PURITY_KEY]

    @cached_property
    def target_category_dict(self):
        ls = [(i - j, j) for i in range(self.max_tot_cn + 1) for j in range(i // 2 + 1)]
        mapped_dict = dict(enumerate(ls))
        return mapped_dict | {
            len(ls): (self.max_tot_cn + 1, self.max_tot_cn // 2 + 1)
        }  # N map

    def __post_init__(self):
        assert (
            self.max_tot_cn <= self.max_tot_cn_arch
        ), "must have max_tot_cn <= max_tot_cn_arch as this has architecture \
            implications"

        self.base_info = SeqInfo(self.max_seq_length, self.prepend_len, postpend_len=1)

        self.key_mapping = {
            key: i for i, key in enumerate(self.supervised_predict_keys)
        }

        # define tensors
        self.poss_haplotypes_tensor = torch.tensor([[0.0, 0], [0, 1], [1, 0], [1, 1]])

        self.seq_dim = 0
        self.glob_dim = 1

        self.surplus_cat = len(self.target_category_dict) - 1
        self.cats = torch.arange(len(self.target_category_dict))

        self.major_vals = torch.tensor(
            [self.target_category_dict[i.item()][0] for i in self.cats]
        )
        # torch.floor((-1 + torch.sqrt(1+8*cats))/2)
        self.minor_vals = torch.tensor(
            [self.target_category_dict[i.item()][1] for i in self.cats]
        )

        self.minor_vals_exp = self.minor_vals.clone()
        self.minor_vals_exp[-1] = 0  # for expectation

        self.one_hot_maj = F.one_hot(self.major_vals).float()
        self.one_hot_min = F.one_hot(self.minor_vals).float()
        self.surplus_cats = torch.tensor(
            [self.max_tot_cn + 1, self.max_tot_cn // 2 + 1]
        )
        self.one_hot_tot = F.one_hot(
            torch.tensor(
                [
                    sum(self.target_category_dict[i.item()])
                    if i != self.surplus_cat
                    else self.max_tot_cn + 1
                    for i in self.cats
                ]
            )
        ).float()
        self.tokens = None

    def to_device(self, device):
        self.poss_haplotypes_tensor = self.poss_haplotypes_tensor.to(device)
        self.major_vals = self.major_vals.to(device)
        self.minor_vals = self.minor_vals.to(device)
        self.minor_vals_exp = self.minor_vals_exp.to(device)
        self.one_hot_maj = self.one_hot_maj.to(device)
        self.one_hot_min = self.one_hot_min.to(device)
        self.one_hot_tot = self.one_hot_tot.to(device)
        self.surplus_cats = self.surplus_cats.to(device)

    @property
    def device(self):
        return self.surplus_cats.device

    def get_tokens(self, curr_input):
        if self.tokens is None or curr_input.shape[:-1] != self.tokens.shape[:-1]:
            self.tokens = torch.full((*curr_input.shape[:-1], 1), SEQ_TOKEN).to(
                curr_input.device
            )
        return self.tokens

    def process_inputs(self, positional_info, inputs, out_kwargs):
        sample_len = out_kwargs["sample_len"]
        positional_info_w_tokens = torch.cat(
            (positional_info, self.get_tokens(positional_info)), dim=-1
        )
        inputs[..., self.prepend_len + sample_len :] = 0
        positional_info_w_tokens[..., self.prepend_len + sample_len :] = 0

        positional_info_w_tokens[
            ...,
            self.prepend_len + sample_len,
            TOKEN_IN_DIM,
        ] = GLOBAL_TOKEN

        return (
            positional_info_w_tokens,
            inputs,
            out_kwargs | {"key_mapping": self.key_mapping},
        )

    def decoder_to_global(self, output, sample_len):
        return output[torch.arange(output.shape[0]), self.prepend_len + sample_len]

    def check_output_vals_align_token(self, positional_info, sample_len):
        assert (
            positional_info[
                torch.arange(positional_info.shape[0]),
                self.prepend_len + sample_len,
                TOKEN_IN_DIM,
            ]
            == GLOBAL_TOKEN
        ).all()

    def process_for_analyses(
        self, positional_info, input, output, sample_len, for_inf=False
    ):
        self.to_device(input.device)  # once on device, no real overhead
        self.check_output_vals_align_token(positional_info, sample_len)

        positional_info, input, seq_output = (
            self.remove_extra(
                val,
                prepend_len=self.prepend_len,
                len_no_postpend=sample_len if for_inf else None,
            )
            # to allow for batches of different sample len, only remove postpend
            # for inference
            for val in [positional_info, input, output[self.seq_dim]]
        )

        # as architecture is fixed, we remove extra categories
        seq_output = seq_output[..., : self.surplus_cat + 1]

        predict_vals = self.decoder_to_global(output[self.glob_dim], sample_len)

        if self.avg_rd_trim_ratio is not None:
            # ovverrides global rd param with avg calc
            predict_vals = self.get_average_read_depth(
                input, seq_output, predict_vals, sample_len
            )

        return {
            "positional_info": positional_info,
            "input": input,
            "output": (seq_output, predict_vals),
            "sample_len": sample_len,
        }

    def get_purity(self, predict_vals):
        purity_idx = self.key_mapping[PURITY_KEY]
        return predict_vals[:, purity_idx : purity_idx + 1]  # keep dim

    def get_allelic_copy_numbers(self, seq_output, predict_vals):
        return 1 + self.get_purity(predict_vals).unsqueeze(-1) * (
            self.get_expected(seq_output) - 1
        )

    def get_total_copy_numbers(self, seq_output, predict_vals):
        purity = self.get_purity(predict_vals)
        total_tumor_cns = self.get_expected(seq_output).sum(axis=-1)
        return total_tumor_cns * purity + (1 - purity) * 2

    def get_robust_vals(self, batch_rd, batch_cn):
        lower_val = torch.quantile(batch_rd, self.avg_rd_trim_ratio, dim=0)
        upper_val = torch.quantile(
            batch_rd, 1 - self.avg_rd_trim_ratio * self.upper_trim_factor, dim=0
        )
        mean_mask = (batch_rd >= lower_val) & (batch_rd <= upper_val)
        return batch_rd[mean_mask], batch_cn[mean_mask]

    def robust_rd_per_cn_mean(self, batch_rd, batch_cn):
        rd, cn = self.get_robust_vals(batch_rd, batch_cn)
        return rd.mean() / cn.mean()

    def get_average_read_depth(self, input, seq_output, predict_vals, sample_len):
        purity = self.get_purity(predict_vals)
        total_cns = self.get_total_copy_numbers(seq_output, predict_vals)
        rd = input[..., READ_IN_DIM]
        mean_rd_per_cn = torch.stack(
            [
                self.robust_rd_per_cn_mean(rd[i, :length], total_cns[i, :length])
                for i, length in enumerate(sample_len)
            ],
            dim=0,
        ).unsqueeze(-1)
        return torch.cat((mean_rd_per_cn, purity), dim=-1)

    def get_supervised_globals(self, output):
        return output[self.glob_dim]

    def get_expected(self, logits, include_surplus=False):
        # python idexing is exclusive
        end_dim = self.surplus_cat + 1 if include_surplus else self.surplus_cat
        probs = F.softmax(logits, dim=-1)
        # do not include TN+ category
        return torch.stack(
            (
                (probs[..., :end_dim] * self.major_vals[:end_dim]).sum(
                    dim=-1
                ),  # 1H dim
                (probs[..., :end_dim] * self.minor_vals_exp[:end_dim]).sum(dim=-1),
            ),
            dim=-1,
        )

    def get_marginal_probs(self, logits):
        # note that this decouples max/min for their probs
        probs = torch.softmax(logits, dim=-1)
        total_probs = probs @ self.one_hot_tot
        major_probs = probs @ self.one_hot_maj
        minor_probs = probs @ self.one_hot_min
        return_probs = torch.zeros(*total_probs.shape, 3)
        return_probs[..., 0] = total_probs
        return_probs[..., : major_probs.shape[-1], 1] = major_probs
        return_probs[..., : minor_probs.shape[-1], 2] = minor_probs
        return return_probs.permute(0, 1, 3, 2)

    def get_category_as_target(self, logits):
        cat = torch.softmax(logits, dim=-1).argmax(-1)
        return torch.stack((self.major_vals[cat], self.minor_vals[cat]), dim=-1)

    def inference(self, pos, input, result, max_len):
        self.to_device(input.device)  # once on device, no real overhead
        out_dict = self.process_for_analyses(pos, input, result, max_len, for_inf=True)
        seq_logits, globs, *_ = out_dict["output"]
        global_dict = {
            f"{key}": globs[..., i]
            for i, key in enumerate(self.supervised_predict_keys)
        }

        copy_numbers = self.get_category_as_target(seq_logits)
        copy_number_probs = self.get_marginal_probs(seq_logits)

        return {
            "positional_info": out_dict["positional_info"],
            "input": out_dict["input"],
            "output": {
                "copy_numbers": copy_numbers,
                "orig_cat_probs": torch.softmax(seq_logits, dim=-1),
                "marginal_probs": copy_number_probs,
            },
            "globals": global_dict,
        }


@dataclass
class UnsupervisedTrainInfo(PairedInfo, TrainTaskInfo):
    read_recon_weight: float = 1e-3
    baf_recon_weight: float = 1e-3

    targets_included = False

    def __post_init__(self):
        super().__post_init__()  # TODO- kinda gross, no longer dc?
        num_cats = len(self.target_category_dict)
        self.penalty_matrix = torch.ones((num_cats, num_cats)) - torch.eye(
            num_cats
        )  # Penalty matrix with ones and zeros on the diagonal

    def to_device(self, device):
        super().to_device(device)
        self.penalty_matrix = self.penalty_matrix.to(device)

    def process_for_loss(self, batch, output):
        positional_info, input, dynamic_kwargs = batch

        return self.process_for_analyses(
            positional_info, input, output, dynamic_kwargs["sample_len"]
        )

    def batch_recon_loss(self, batch_read_in, batch_cn, read_depth_out):
        robust_rd_in, robust_cn = self.get_robust_vals(batch_read_in, batch_cn)
        return F.l1_loss(
            read_depth_out * robust_cn, robust_rd_in, reduction="none"
        ).mean(axis=-1)

    def read_recon_loss(self, input, total_cns, read_depth_out, sample_len):
        read_in = input[..., READ_IN_DIM]
        return torch.stack(
            [
                self.batch_recon_loss(
                    read_in[i, :length], total_cns[i, :length], read_depth_out[i]
                )
                for i, length in enumerate(sample_len)
            ],
            dim=0,
        )

    def baf_recon_loss(self, input, out_cns, sample_len):
        total_out = out_cns.sum(dim=-1)

        poss_minor_c = torch.einsum("hd,bnd->bnh", self.poss_haplotypes_tensor, out_cns)

        poss_baf = torch.zeros_like(poss_minor_c)
        mask = total_out != 0
        poss_baf[mask] = poss_minor_c[mask] / total_out[mask].unsqueeze(-1)

        baf_in = input[..., BAF_IN_DIM]

        baf_in_reshape = baf_in.unsqueeze(-1).repeat(
            1, 1, self.poss_haplotypes_tensor.shape[0]
        )

        most_likely_snp_baf_recon, _ = F.mse_loss(
            poss_baf, baf_in_reshape, reduction="none"
        ).min(dim=-1)

        return get_stacked_batch_loss(most_likely_snp_baf_recon, sample_len)

    def recon_loss(self, input, output, sample_len, **_):
        read_depth = output[self.glob_dim][:, self.key_mapping[READ_KEY]]
        allelic_cns = self.get_allelic_copy_numbers(*output)
        return (
            self.read_recon_loss(
                input, allelic_cns.sum(dim=-1), read_depth, sample_len
            ),
            self.baf_recon_loss(input, allelic_cns, sample_len),
        )

    def reconstruction_loss_detail(
        self,
        *args,
        prefix="train",
        **kwargs,
    ):
        # RECON losses are unsupervised
        read_loss, baf_loss = self.recon_loss(*args, **kwargs)
        return {
            f"{prefix}/read_recon": read_loss,
            f"{prefix}/baf_recon": baf_loss,
        }

    def default_loss(self, input, output, sample_len, **_):
        loss_tuple = self.recon_loss(input, output, sample_len)

        weights_tuple = (
            self.read_recon_weight,
            self.baf_recon_weight,
        )

        return sum(
            loss * loss_weights for loss, loss_weights in zip(loss_tuple, weights_tuple)
        ).mean()


@dataclass
class SupervisedTrainInfo(UnsupervisedTrainInfo, TrainTaskInfo):
    read_recon_weight: float = 1e-3
    baf_recon_weight: float = 1e-3

    supervised_read_recon: bool = False

    base_info: TrainSeqInfo = field(init=False)  # overwrite
    targets_included = True

    def __post_init__(self):
        super().__post_init__()  # TODO- kinda gross, no longer dc?
        self.base_info = TrainSeqInfo(
            self.max_seq_length, self.prepend_len, postpend_len=1
        )
        self.loss_weights = self.loss_weights or [
            1 for _ in self.supervised_predict_keys
        ]
        assert len(self.loss_weights) == len(self.supervised_predict_keys)

    def process_inputs(self, positional_info, inputs, targets, out_kwargs, global_info):
        positional_info, inputs, extra_info = super().process_inputs(
            positional_info, inputs, out_kwargs
        )

        return (positional_info, inputs, targets, out_kwargs, global_info | extra_info)

    def process_targets(self, targets, sample_len_for_inf=None):
        return self.remove_extra(
            targets,
            prepend_len=self.prepend_len,
            len_no_postpend=sample_len_for_inf,
        )

    def process_for_loss(self, batch, output):
        positional_info, input, targets, dynamic_kwargs, global_info = batch

        targets = self.process_targets(targets)

        return (
            {"targets": targets}
            | self.process_for_analyses(
                positional_info, input, output, dynamic_kwargs["sample_len"]
            )
            | {"global_info": global_info}
        )

    def supervised_loss(self, output, global_info, **_):
        out_globs = self.get_supervised_globals(output)
        stacked_targets = torch.stack(
            [global_info[key] for key in self.supervised_predict_keys], dim=-1
        )
        return F.l1_loss(out_globs, stacked_targets, reduction="none").T

    def supervised_loss_detail(self, output, global_info, prefix="train", **_):
        # optional metric
        supervised_loss = self.supervised_loss(output, global_info)
        return {
            f"{prefix}/{key}": supervised_loss[i]
            for i, key in enumerate(self.supervised_predict_keys)
        }

    def get_target_as_category(self, targets):
        # this efficiently creates the correct mapping for the rarget dict
        targs = torch.round(targets).long()
        total = targs.sum(-1)
        return torch.clip(
            torch.ceil(total / 2) * torch.ceil((total + 1) / 2) + targets[:, 1],
            max=self.surplus_cat,
        ).long()

    def sequence_loss(self, output, targets, sample_len, **_):
        paired_logits = output[self.seq_dim]
        paired_logits = torch.flatten(paired_logits, end_dim=1)
        targs = torch.flatten(targets, end_dim=1)
        cats = self.get_target_as_category(targs)
        loss = F.cross_entropy(paired_logits, cats, reduction="none").reshape(
            targets.shape[0:2]
        )  # cross_entropy expects long
        return get_stacked_batch_loss(loss, sample_len)

    def recon_loss(self, input, output, targets, sample_len, global_info, **_):
        read_depth = (
            global_info[READ_KEY]
            if self.supervised_read_recon
            else output[self.glob_dim][:, self.key_mapping[READ_KEY]]
        )
        allelic_cns = self.get_allelic_copy_numbers(
            output[self.seq_dim], output[self.glob_dim]
        )
        # 0 reconstruction for targets that exceed the max considered
        mask = targets.sum(-1, keepdim=True) <= self.max_tot_cn
        masked_input, masked_allelic_cns = (
            torch.where(mask, input, 0),
            torch.where(mask, allelic_cns, 0),
        )
        return (
            self.read_recon_loss(
                masked_input,
                masked_allelic_cns.sum(dim=-1),
                read_depth.unsqueeze(1),
                sample_len,
            ),
            self.baf_recon_loss(masked_input, masked_allelic_cns, sample_len),
        )

    def default_loss(self, input, output, targets, sample_len, global_info, **_):
        loss_tuple = (
            self.sequence_loss(output, targets, sample_len),
            *self.supervised_loss(output, global_info),
            *self.recon_loss(input, output, targets, sample_len, global_info),
        )

        weights_tuple = (
            1,
            *self.loss_weights,
            self.read_recon_weight,
            self.baf_recon_weight,
        )

        return sum(
            loss * loss_weights for loss, loss_weights in zip(loss_tuple, weights_tuple)
        ).mean()


    def discrete_accuracy(self, output, targets, sample_len, **_):
        inf = self.get_category_as_target(output[self.seq_dim])
        targs = torch.where(
            targets.sum(-1, keepdim=True) <= self.max_tot_cn, targets, self.surplus_cats
        )

        inf_w_tot = torch.cat((inf, inf.sum(-1, keepdim=True)), -1)
        targs_w_tot = torch.cat((targs, targs.sum(-1, keepdim=True)), -1)

        loss = torch.eq(torch.round(inf_w_tot), targs_w_tot).float()

        return get_stacked_batch_loss(loss, sample_len)
