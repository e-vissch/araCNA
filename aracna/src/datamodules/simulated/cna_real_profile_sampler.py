import os

import numpy as np
import pandas as pd
from aracna.src.datamodules.simulated.cna_sampler import ProfileSampler
from aracna.src.task_info.task_info import SeqInfo
from aracna.src.utils.constants import AVG_SNP_DIST


def write_merged_profiles(data_path, out_fname):
    summary_df = pd.read_csv(
        f"{data_path}/summary.ascatv3TCGA.penalty70.hg38.tsv", index_col=0, sep="\t"
    )
    segment_data = []
    for segment_file in os.listdir(f"{data_path}/segments"):
        df = pd.read_csv(f"{data_path}/segments/{segment_file}", sep="\t")
        segment_data.append(df)
    segment_df = pd.concat(segment_data)
    segment_df["segment_length"] = segment_df["endpos"] - segment_df["startpos"]
    segment_df["totalC"] = segment_df["nMajor"] + segment_df["nMinor"]

    SELECT_COLS = [
        "sample",
        "patient",
        "cancer_type",
        "sex",
        "barcodeTumour",
        "barcodeNormal",
        "purity",
        "ploidy",
    ]
    merged_df = pd.merge(
        segment_df,
        summary_df.reset_index().rename(columns={"name": "sample"})[SELECT_COLS],
        on="sample",
    )
    merged_df.to_csv(f"{data_path}/{out_fname}", index=False, compression="gzip")


def load_cna_data(
    data_path, out_fname="merged_profiles.csv.gz", group_col="patient", target_split=0.8
):
    if not os.path.exists(f"{data_path}/{out_fname}"):
        write_merged_profiles(data_path, out_fname)

    merged_df = pd.read_csv(f"{data_path}/{out_fname}", compression="gzip")
    merged_df = merged_df.sort_values(by=["sample", "chr", "startpos"])
    # depending on the group_col, total train/split might not be target_split

    unique_samples = merged_df["sample"].unique()
    val_groups = np.random.choice(
        unique_samples, int(len(unique_samples) * (1 - target_split)), replace=False
    )
    train_df = merged_df[~merged_df[group_col].isin(val_groups)]
    val_df = merged_df[merged_df[group_col].isin(val_groups)]
    return train_df, val_df


class RealProfileMixin:
    @classmethod
    def real_from_config(cls, task_info, train_df, val_df, dataset_kwargs):
        raise NotImplementedError

    @classmethod
    def from_path(cls, task_info, dataset_kwargs, **child_kwargs):
        copy_dataset_kwargs = dataset_kwargs.copy()
        data_path = copy_dataset_kwargs.pop("data_path")
        train_df, val_df = load_cna_data(data_path)
        return cls.real_from_config(
            train_df, val_df, task_info, copy_dataset_kwargs, **child_kwargs
        )


class RealProfileSampler(ProfileSampler, RealProfileMixin):
    def __init__(
        self,
        train_df,
        val_df,
        task_info: SeqInfo,
        sample_col="sample",
        read_depth_range=(15, 15),
        baf_scale_range=(1, 1),
    ) -> None:
        super().__init__(task_info, read_depth_range, baf_scale_range)

        self.df_dict = {"train": train_df, "val": val_df}
        self.sample_col = sample_col

    def _get_sample(self, df):
        # sample a measured sample
        sample_id = np.random.choice(df[self.sample_col].unique())
        return (df[df[self.sample_col] == sample_id]).copy()

    def _sample_snps_from_segments(self, sub_df):
        # p.random.choice slow, so just get approx. segment length/avg_snp_dist
        # no. of snps (approx. as we might sample the same snp twice, hence np.unique)

        snp_loc_ls = [
            np.unique(
                np.random.randint(
                    low=start_pos,
                    high=end_pos,
                    size=np.ceil(segment_length / AVG_SNP_DIST).astype(int),
                )
            )
            for start_pos, end_pos, segment_length in sub_df[
                ["startpos", "endpos", "segment_length"]
            ].values
        ]
        seg_lens = [sampled_snps.shape[0] for sampled_snps in snp_loc_ls]

        return np.concatenate(snp_loc_ls), seg_lens

    def _repeat_profiles_for_segments(self, sub_df, seg_lens):
        sub_df["sequence_snp_len"] = seg_lens
        # build n_parental, by repeating nMajor, nMinor for each snp in segment
        n_parental = np.concatenate(
            [
                np.repeat([[n_major, n_minor]], segment_snp_len, axis=0)
                for n_major, n_minor, segment_snp_len in sub_df[
                    ["nMajor", "nMinor", "sequence_snp_len"]
                ].values
            ]
        )

        # get chromosome profiles for positional encodings
        chr_vals = np.concatenate(
            [
                np.repeat(self.chrom_mapping[chr], chr_count)
                for chr, chr_count in sub_df.groupby("chr")["sequence_snp_len"]
                .sum()
                .items()
            ]
        )
        return n_parental, chr_vals

    def sample_new_batch(self, train_type):
        self.zero_sample_arrays()  # reset arrays for sampling

        sub_df = self._get_sample(self.df_dict[train_type])
        sampled_snp_locs, seg_lens = self._sample_snps_from_segments(sub_df)
        self.n_segments = len(seg_lens)
        sampled_n_parental, sampled_chr_vals = self._repeat_profiles_for_segments(
            sub_df, seg_lens
        )
        sampled_seq_len = self._fit_seqs_to_arr(
            sampled_snp_locs, sampled_n_parental, sampled_chr_vals
        )

        self.start_seq, self.end_seq = self.get_start_end_seq(sampled_seq_len)

        self.target_info = (
            sub_df[["purity", "ploidy", "cancer_type"]]
            .drop_duplicates()
            .to_dict(orient="records")[0]
        )  # only one row

    @classmethod
    def real_from_config(cls, task_info, train_df, val_df, dataset_kwargs):
        return cls(train_df, val_df, task_info, **dataset_kwargs)
