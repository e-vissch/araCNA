from dataclasses import dataclass
import torch 
from aracna.src.task_info.cat_paired import PairedInfo, SupervisedTrainInfo, UnsupervisedTrainInfo
from aracna.src.utils.constants import (
    PLOIDY_KEY,
    PURITY_KEY,
    READ_IN_DIM,
)
import torch.nn.functional as F


@dataclass
class PairedLogRInfo(PairedInfo):

    avg_rd_trim_ratio = None
    
    @property
    def supervised_predict_keys(self):
        return [PLOIDY_KEY, PURITY_KEY]
    
    
    def get_ploidy_purity(self, seq_output, predict_vals, sample_len):
        purity = self.get_purity(predict_vals)
        total_cns = self.get_total_copy_numbers(seq_output, predict_vals)
        ploidy = total_cns[:, :sample_len].mean(-1, keepdims=True)
        return torch.cat((ploidy, purity), dim=-1)


    def process_for_analyses(self, positional_info, input, output, sample_len, for_inf=False):
        return_dict = super().process_for_analyses(positional_info, input, output, sample_len, for_inf)

        seq_output, predict_vals = return_dict["output"]

        predict_vals = self.get_ploidy_purity(
                seq_output, predict_vals, return_dict["sample_len"]
            )

        return_dict["output"] = seq_output, predict_vals
        return return_dict



@dataclass
class UnsupLogR(PairedLogRInfo, UnsupervisedTrainInfo):
    def batch_recon_loss(self, batch_log_read_in, batch_cn):
        robust_log_rd_in, robust_cn = self.get_robust_vals(batch_log_read_in, batch_cn)
        return F.l1_loss(
            torch.log2(robust_cn/robust_cn.mean() + 1e-6), robust_log_rd_in, reduction="none"
        ).mean(axis=-1)
    

    def read_recon_loss(self, input, total_cns, sample_len):
        read_in = input[..., READ_IN_DIM]
        return torch.stack(
            [
                self.batch_recon_loss(
                    read_in[i, :length], total_cns[i, :length], 
                )
                for i, length in enumerate(sample_len)
            ],
            dim=0,
        )


    def recon_loss(self, input, output, sample_len, **_):
        allelic_cns = self.get_allelic_copy_numbers(*output)
        return (
            self.read_recon_loss(
                input, allelic_cns.sum(dim=-1), sample_len
            ),
            self.baf_recon_loss(input, allelic_cns, sample_len),
        )


@dataclass
class SupLogR(UnsupLogR, SupervisedTrainInfo):
    # still need to define for inheritanvce
    
    def recon_loss(self, input, output, targets, sample_len, **_):
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
                sample_len,
            ),
            self.baf_recon_loss(masked_input, masked_allelic_cns, sample_len),
        )

