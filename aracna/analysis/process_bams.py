import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
import pysam


def check_read_passes(read, min_mapq=35):
    if (
        read.is_unmapped
        or read.is_secondary
        or read.is_qcfail
        or read.is_duplicate
        or read.is_supplementary
        or read.mapping_quality < min_mapq
    ):
        return False  # Exclude this read
    return True  # Include this r


def init_worker(eg_bam):
    global bam
    bam = pysam.AlignmentFile(eg_bam, "rb")


def process_row(index, snp_dataframe, window_val):
    # add min_counts exclude criteria
    row = snp_dataframe.iloc[index]
    # Open a new BAM file handle per process

    chr_val = f"chr{row.chr}" if row.chr != 23 else "chrX"
    coverage_snp = bam.count_coverage(
        contig=chr_val,
        start=row.position - 1,
        stop=row.position,
        quality_threshold=20,
        read_callback=check_read_passes,
    )
    coverage_window = bam.count_coverage(
        contig=chr_val,
        start=row.position - window_val - 1,
        stop=row.position + window_val,
        quality_threshold=0,
        read_callback=check_read_passes,
    )

    snp_pos_cov = np.array(coverage_snp)[:, 0]
    coverage_window = np.array(coverage_window)

    # Process the coverage data as before
    depth = snp_pos_cov[row.a1 - 1] + snp_pos_cov[row.a0 - 1]
    # row.a1 is index of minor allele
    BAF = np.nan if depth == 0 else snp_pos_cov[row.a1 - 1] / depth

    cov = np.sum(coverage_window, axis=0)
    zero_count = cov[cov < 1e-6].shape[0]

    windowed_depth = np.mean(np.sum(coverage_window, axis=0))
    return index, depth, windowed_depth, BAF, zero_count


def write_full_df(loci_file, snp_dir_stub, write_file):
    # only once for preproccessing
    loci = pd.read_csv(loci_file, sep="\t")
    loci = loci.rename(columns={"Chr": "chr", "Position": "position"})
    loci["chr"] = (
        loci["chr"].str.replace("chr", "").replace({"X": 23, "Y": 24}).astype(int)
    )
    concat_ls = []
    for chr_val in [*list(range(1, 23)), "X"]:
        allele_file = pd.read_csv(f"{snp_dir_stub}chr{chr_val}.txt", sep="\t")
        act_chr_val = chr_val if chr_val != "X" else 23
        concat_ls.append(
            pd.merge(allele_file, loci[loci.chr == act_chr_val], on="position")
        )
    full_df = pd.concat(concat_ls)
    full_df.position = full_df.position.astype(int)

    full_df[["chr", "position", "a0", "a1"]].to_csv(write_file, index=False)


def check_index(eg_bam):
    bam = pysam.AlignmentFile(eg_bam, "rb")
    if bam.check_index():
        print("BAM file is already indexed.")
    else:
        print("indexing bam file")
        pysam.index(eg_bam)


def get_file_info(bam_file, snp_file, out_file, num_processes, window_val=1000):
    chunksize = 100

    snp_dataframe = pd.read_csv(snp_file)

    check_index(bam_file)

    with multiprocessing.Pool(
        processes=num_processes, initializer=init_worker, initargs=[bam_file]
    ) as pool:
        results = pool.map(
            partial(process_row, snp_dataframe=snp_dataframe, window_val=window_val),
            snp_dataframe.index,
            chunksize=chunksize,
        )

    id_cols = ["chr", "position"]
    info_cols = ["depth", "windowed_depth", "BAF", "windowed_zero_count"]

    results_df = pd.DataFrame(results, columns=["index_val", *info_cols]).sort_values(
        "index_val"
    )
    results_df[id_cols] = snp_dataframe[id_cols]
    results_df = results_df.dropna()
    results_df[id_cols + info_cols].to_csv(out_file, index=False)
