from math import ceil

import numpy as np
from lightning.pytorch.callbacks import Callback
from aracna.src.datamodules.simulated.main_cna_sampling_func import (
    sample_cnas_from_input_ranges,
)
from aracna.src.datamodules.simulated.sample_datamodule import BaseSampler
from aracna.src.task_info.task_info import SeqInfo
from aracna.src.utils.constants import CHROM_SIZE_DICT, N_CHROM

sampling_func_register = {"paired": sample_cnas_from_input_ranges}


class ProfileSampler(BaseSampler):
    def __init__(
        self,
        task_info: SeqInfo,
        read_depth_range=(15, 15),
        read_depth_scale_range=(0.5, 0.5),
        baf_scale_range=(0.05, 0.05),
        purity_range=(1, 1),
        sampling_name=None,
        _start_seqlen=None,
    ) -> None:
        self.task_info = task_info
        # for annealing sequnce length, can change based on training schedule
        self.read_depth_range = read_depth_range
        self.read_depth_scale_range = read_depth_scale_range
        self.baf_scale_range = baf_scale_range
        self.purity_range = purity_range

        self.chrom_mapping = {str(i): i for i in range(1, N_CHROM)} | {"X": N_CHROM}

        # built over sampled chroms.
        self.n_segments = 0
        self.start_seq, self.end_seq = None, None

        self.sampling_func = sampling_func_register.get(
            sampling_name, sample_cnas_from_input_ranges
        )
        self.sample_kwargs = {}
        self.set_curr_seqlen(_start_seqlen or self.task_info.max_fillable_seq_length)

    @property
    def curr_seqlen_w_extra(self):
        return (
            self.curr_seqlen + self.task_info.prepend_len + self.task_info.postpend_len
        )

    def init_sample_arrays(self):
        # initialsize all arrays:
        self.snp_locs = np.zeros(self.curr_seqlen_w_extra)
        self.n_parental = np.zeros((self.curr_seqlen_w_extra, 2))
        self.chr_vals = np.zeros(self.curr_seqlen_w_extra)

        self.reads = np.zeros(self.curr_seqlen_w_extra)
        self.minor_allele_freq_meas = np.zeros(self.curr_seqlen_w_extra)
        self.major_parental = np.zeros(self.curr_seqlen_w_extra)
        self.minor_parental = np.zeros(self.curr_seqlen_w_extra)

    def zero_sample_arrays(self):
        self.snp_locs.fill(0)
        self.n_parental.fill(0)
        self.chr_vals.fill(0)

    def zero_data_arrays(self):
        self.reads.fill(0)
        self.minor_allele_freq_meas.fill(0)
        self.major_parental.fill(0)
        self.minor_parental.fill(0)

    @property
    def max_seq_length(self):
        return self.task_info.max_seq_length

    def set_curr_seqlen(self, curr_seqlen):
        self.curr_seqlen = min(
            curr_seqlen - self.task_info.postpend_len,
            self.task_info.max_fillable_seq_length,
        )
        self.init_sample_arrays()

    def get_start_end_seq(self, sampled_seq_length):
        return (
            self.task_info.prepend_len,
            sampled_seq_length + self.task_info.prepend_len,
        )

    def _fit_seqs_to_arr(self, snp_locs_, n_parental_, chr_vals_):
        if (sampled_seq_length := snp_locs_.shape[0]) > self.curr_seqlen:
            # Fast way to randomly select
            mask = np.zeros(sampled_seq_length, dtype=bool)
            mask[: self.curr_seqlen] = True
            np.random.shuffle(mask)

            # Apply the mask to each array
            snp_locs_ = snp_locs_[mask]
            n_parental_ = n_parental_[mask]
            chr_vals_ = chr_vals_[mask]

            sampled_seq_length = self.curr_seqlen

        start_seq, end_seq = self.get_start_end_seq(sampled_seq_length)
        self.snp_locs[start_seq:end_seq] = snp_locs_
        self.n_parental[start_seq:end_seq] = n_parental_
        self.chr_vals[start_seq:end_seq] = chr_vals_

        return sampled_seq_length

    def _sample_read_params(self):
        read_depth = round(np.random.uniform(*self.read_depth_range), 2)
        read_depth_scale = (
            round(np.random.uniform(*self.read_depth_scale_range), 2) * read_depth
        )
        baf_scale = round(np.random.uniform(*self.baf_scale_range), 3)
        purity = round(np.random.uniform(*self.purity_range), 2)
        return read_depth, read_depth_scale, baf_scale, purity

    def sample_new_batch(self, train_type):
        # should set n_parental, snp/chrom inputs and start_seq, end_seq
        raise NotImplementedError

    def write_data_to_arrays(self, output_data):
        (
            self.reads[self.start_seq : self.end_seq],
            self.minor_allele_freq_meas[self.start_seq : self.end_seq],
            self.major_parental[self.start_seq : self.end_seq],
            self.minor_parental[self.start_seq : self.end_seq],
        ) = output_data

    def sample(self, train_type):
        # TODO, may be a better way of doing this w batch size,
        # with pytorch lightning it automatically handles so have to do this.

        self.zero_data_arrays()  # reset arrays for sampling

        self.sample_new_batch(train_type)

        input_params, output_data = self.sampling_func(
            self.n_parental[self.start_seq : self.end_seq],
            self.read_depth_range,
            self.read_depth_scale_range,
            self.baf_scale_range,
            self.purity_range,
            **self.sample_kwargs,
        )

        read_depth, read_depth_scale, baf_scale, purity = input_params

        self.write_data_to_arrays(output_data)

        input_info = {
            "read_depth": read_depth,
            "baf_scale": baf_scale,
            "purity": purity,
            "read_depth_scale": read_depth_scale,
        }

        return (
            (self.snp_locs, self.chr_vals),  # positional info
            (self.reads, self.minor_allele_freq_meas),  # inputs
            (self.major_parental, self.minor_parental),  # targets
            {"sample_len": self.end_seq - self.start_seq},
            input_info,
        )


class PurelySimulated(ProfileSampler):
    def __init__(
        self,
        task_info: SeqInfo,
        read_depth_range=(15, 15),
        read_depth_scale_range=(0.5, 0.5),
        baf_scale_range=(0.05, 0.05),
        purity_range=(1, 1),
        sampling_name=None,
        _start_seqlen=None,
        max_total=10,
        small_segment_threshold=10,
        small_segment_swap_threshold=0.5,
        sample_seqlen=False,
        min_segment_length=100,
        inject_homoz_loci=True,
    ) -> None:
        super().__init__(
            task_info,
            read_depth_range,
            read_depth_scale_range,
            baf_scale_range,
            purity_range,
            sampling_name,
            _start_seqlen,
        )

        self.small_segment_threshold = small_segment_threshold
        self.small_segment_swap_threshold = small_segment_swap_threshold
        self.sample_seqlen = sample_seqlen
        self.start_seq = task_info.prepend_len
        self.end_seq = 0  # cumulatively set
        self.min_segment_length = min_segment_length

        self.sample_kwargs = {"inject_homoz_loci": inject_homoz_loci}

        self.set_max_total_and_profiles(max_total)

        # set in sample_n_segments
        self.approx_total_segments = None
        self.prob_factors = None

    def set_max_total_and_profiles(self, max_total):
        self.max_total = max_total
        self.sample_kwargs |= {"max_total": max_total}
        self.poss_profiles = [
            (i, j)
            for i in range(max_total + 1)
            for j in range(min(max_total - i, i) + 1)
        ]

    def get_min_length_change_pts(self, segment_change_pts):
        valid_change_pts = [segment_change_pts[0]]
        for pt in segment_change_pts[1:]:
            if pt - valid_change_pts[-1] >= self.min_segment_length:
                valid_change_pts.append(pt)
        return valid_change_pts

    def sample_segments(self, seq_len, approx_n_segments):
        segment_change_pts = np.sort(
            np.unique(
                np.random.choice(
                    range(1, seq_len), approx_n_segments - 1, replace=False
                )
            )
        )
        valid_chg_pts = (
            self.get_min_length_change_pts(segment_change_pts)
            if approx_n_segments > 1
            else []
        )

        # Split the target number at the partition points
        segment_lengths = np.diff([0, *list(valid_chg_pts), seq_len])

        n_segments = len(segment_lengths)

        segment_ids = np.random.choice(
            len(self.poss_profiles), p=self.prob_factors, size=n_segments
        )
        segment_vals = np.array([self.poss_profiles[i] for i in segment_ids])

        return segment_vals, segment_lengths, n_segments

    def sample_chrom(self, chrom, chrom_weight, approx_total_segments):
        size_factor = chrom_size = CHROM_SIZE_DICT[chrom] * chrom_weight

        upper_size = int(self.curr_seqlen * size_factor)
        approx_chrom_seqlen = upper_size

        if self.sample_seqlen:
            # sample seq length so that tot is random
            approx_chrom_seqlen = np.random.randint(upper_size * 0.7, upper_size)

        approx_chrom_segments = int(ceil(size_factor * approx_total_segments))

        # way quicker than np.random.choice, might have repeated,
        # hence unique/actual_chrom_seqlen, also hence size being larger to
        # make diff between actual and approx minimial
        snp_locs = np.sort(
            np.unique(
                np.random.randint(
                    low=0, high=chrom_size * 1e6, size=int(approx_chrom_seqlen * 1.5)
                )
            )[:approx_chrom_seqlen]
        )
        actual_chrom_seqlen = snp_locs.shape[0]

        segment_vals, segment_lengths, chrom_segments = self.sample_segments(
            actual_chrom_seqlen, approx_chrom_segments
        )
        self.n_segments += chrom_segments

        # arbitrary binomial params, gave reasonable CNs
        n_parental = np.repeat(
            segment_vals,
            segment_lengths,
            axis=0,
        )

        self.chr_vals[self.end_seq : self.end_seq + actual_chrom_seqlen] = chrom
        self.snp_locs[self.end_seq : self.end_seq + actual_chrom_seqlen] = snp_locs
        self.n_parental[self.end_seq : self.end_seq + actual_chrom_seqlen] = n_parental

        self.end_seq += actual_chrom_seqlen

    def sample_chrom_list(self):
        chrom_list = list(range(1, N_CHROM + 1))
        np.random.shuffle(chrom_list)

        n_chroms = np.random.choice(ceil(23 * self.curr_seqlen / 1e6)) + 1

        return chrom_list[:n_chroms]

    def sample_n_segments(self, chrom_ls):
        if np.random.rand() < self.small_segment_swap_threshold:
            approx_total_segments = np.random.randint(
                len(chrom_ls), max(len(chrom_ls) + 1, self.curr_seqlen / 1e3, 50)
            )
        else:
            # skew distribution towards smaller segments
            approx_total_segments = len(chrom_ls) + np.random.poisson(3)

        prob_factors = prob_factors = [
            1 / (1 + (max(0, (i + j) - 4)) * abs((i + 1) / 8))
            for (i, j) in self.poss_profiles
        ]  # heuristic, equal if total less<=4 else, slowly decrease prob

        # prob_factors = [1 for _ in range(len(self.poss_profiles))]

        if approx_total_segments <= self.small_segment_threshold:
            # skew towards smaller deviations from normal profile & smaller tot CNs
            include_tot_diff = approx_total_segments - 1
            prob_factors = [
                (
                    1 / (1 + td * abs(i + 1))
                    if (td := abs(i - 1) + abs(j - 1)) <= include_tot_diff
                    else 0
                )
                for (i, j) in self.poss_profiles
            ]
            prob_factors[0] = prob_factors[3]  # kinda gros to make symm

        self.prob_factors = prob_factors / np.sum(prob_factors)
        return approx_total_segments

    def sample_new_batch(self, _):
        self.zero_sample_arrays()
        self.end_seq = 0

        chrom_sub = self.sample_chrom_list()
        approx_total_segments = self.sample_n_segments(chrom_sub)

        chrom_weight = 1 / sum(CHROM_SIZE_DICT[chr] for chr in chrom_sub)
        for chrom in chrom_sub:
            self.sample_chrom(chrom, chrom_weight, approx_total_segments)


class SimulatedWarmupDifficulty(Callback):
    def __init__(
        self, sampling_warmup_interval=None, start_tot=4, max_total=10, include=True
    ):
        self.sampling_warmup_interval = sampling_warmup_interval
        self.start_tot = start_tot
        self.max_total = max_total

        self.include = include

    def on_train_epoch_start(self, trainer, pl_module):
        if self.include:
            new_max_total = min(
                self.start_tot + trainer.current_epoch // self.sampling_warmup_interval,
                self.max_total,
            )

            profile_sampler = trainer.datamodule.sampler.profile_sampler
            if new_max_total != profile_sampler.max_total:
                profile_sampler.set_max_total_and_profiles(new_max_total)
