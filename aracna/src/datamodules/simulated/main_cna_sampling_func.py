import numpy as np
import torch


def sample_artifacts(sequence_length, min_length=5, max_length=500, avg_interval=1e3):
    num_segments = np.random.randint(1, max(2, sequence_length // avg_interval))

    # Randomly choose start positions for the segments
    starts = np.random.randint(0, sequence_length - min_length, size=num_segments)

    # Randomly determine segment lengths (at least min_length)
    lengths = np.random.randint(
        min_length, max(min_length + 1, max_length), size=num_segments
    )

    ends = starts + lengths

    return starts, ends, num_segments


def inject_homozygosity_artifacts(minor_allele_freq_meas, min_length=5, max_length=500):
    sequence_length = minor_allele_freq_meas.shape[0]

    starts, ends, num_segments = sample_artifacts(
        sequence_length, min_length, max_length
    )

    homozygous_loci = np.zeros_like(minor_allele_freq_meas).astype(bool)
    # Mark the segments in the modification array
    for i in range(num_segments):
        minor_allele_freq_meas[starts[i] : ends[i]] = np.random.choice(
            [0, 1], size=min(sequence_length, ends[i]) - starts[i]
        )
        homozygous_loci[starts[i] : ends[i]] = True

    return minor_allele_freq_meas, homozygous_loci


def sample_baf(
    reads, minor_allele_freq, snps, inject_homoz_loci, baf_scale, base_df=15
):
    allele_ct = snps.sum(axis=1)
    homozygous_mask = (allele_ct == 0) | (allele_ct == 2) | inject_homoz_loci
    heterozygous_mask = (~homozygous_mask) & (
        (np.abs(minor_allele_freq - 1) < 1e-6) | (np.abs(minor_allele_freq) < 1e-6)
    )

    num_reads = np.random.poisson(reads)
    num_ma = np.random.binomial(num_reads, np.clip(minor_allele_freq, 0, 1))
    maf = np.divide(
        num_ma, num_reads, where=num_reads != 0, out=np.zeros_like(minor_allele_freq)
    )  # 0/0 = 0
    df = max(2, base_df * 10 * baf_scale)

    # Base noise for heterozygous loci
    baf_noise = 0.2 * baf_scale * np.random.standard_t(df=df, size=maf.shape)

    baf_noise[heterozygous_mask] = baf_scale * np.random.standard_t(
        df=df, size=np.sum(heterozygous_mask)
    )

    minor_allele_freq_meas = np.clip(
        np.where(
            (maf + baf_noise <= 1) & (maf + baf_noise >= 0),
            maf + baf_noise,
            maf - baf_noise,
        ),
        0,
        1,
    )

    return minor_allele_freq_meas


def sample_read_params(
    read_depth_range, read_depth_scale_range, baf_scale_range, purity_range
):
    read_depth = round(np.random.uniform(*read_depth_range), 2)
    read_depth_scale = round(np.random.uniform(*read_depth_scale_range), 2) * read_depth
    baf_scale = round(np.random.uniform(*baf_scale_range), 3)
    purity = round(np.random.uniform(*purity_range), 2)
    return read_depth, read_depth_scale, baf_scale, purity


def sample_cnas_from_parental(
    n_parental,
    read_depth,
    read_depth_scale=0.5,
    baf_scale=0.05,
    purity=1,
    inject_homoz_loci=True,
    max_total=10,
):
    """Main sampling logic/routine for generating CNA data from parental profile."""
    sequence_length = n_parental.shape[0]
    minor_copy_number = np.min(n_parental, axis=1)

    sample_parental = n_parental

    total_c = purity * sample_parental.sum(axis=1).astype(int) + 2 * (1 - purity)
    snps = np.random.binomial(n=1, p=0.5, size=(sequence_length, 2))
    minor_c = (purity * snps * sample_parental + (1 - purity) * snps).sum(axis=1)
    minor_allele_freq = np.divide(
        minor_c, total_c, where=total_c != 0, out=np.zeros_like(minor_c)
    )  # 0/0 = 0

    sample_total_c = total_c

    reads = np.clip(
        sample_total_c * read_depth
        + read_depth_scale * np.random.standard_t(df=2, size=sequence_length),
        0,
        None,
    )

    if inject_homoz_loci:
        minor_allele_freq, inject_homoz_loci = inject_homozygosity_artifacts(
            minor_allele_freq
        )
    else:
        inject_homoz_loci = np.zeros_like(minor_allele_freq).astype(bool)

    minor_allele_freq_meas = sample_baf(
        reads,
        minor_allele_freq,
        snps,
        inject_homoz_loci,
        baf_scale,
    )

    return (
        reads,
        minor_allele_freq_meas,
        np.max(n_parental, axis=1),
        minor_copy_number,
    )


def sample_cnas_from_input_ranges(
    n_parental,
    read_depth_range,
    read_depth_scale_range,
    baf_scale_range,
    purity_range,
    **kwargs,
):
    input_params = sample_read_params(
        read_depth_range, read_depth_scale_range, baf_scale_range, purity_range
    )
    output_params = sample_cnas_from_parental(n_parental, *input_params, **kwargs)
    return input_params, output_params


def get_input_profile(short_profile, sub_seqlen):
    n_segs = short_profile.shape[1]
    seg_len = sub_seqlen // n_segs
    parental = np.zeros((sub_seqlen, 2))

    for i in range(n_segs):
        next_idx = (i + 1) * seg_len if i < n_segs - 1 else sub_seqlen
        parental[i * seg_len : next_idx, :] = short_profile[:, i]
    return parental


def pos_vals(sub_seqlen, start_pos=1000, avg_dist=1.7e3):
    return np.arange(start_pos, int(sub_seqlen * avg_dist) + start_pos, int(avg_dist))


def sample_from_profile(
    short_profile,
    sub_seqlen,
    total_seqlen,
    read_depth,
    purity,
    rd_scale_range,
    baf_scale_range,
    int_dtype=torch.int32,
    float_dtype=torch.float32,
    sample_func=sample_cnas_from_input_ranges,
):
    input_tensor = np.zeros((1, total_seqlen, 2))
    pos_info = np.zeros((1, total_seqlen, 2), dtype=int)
    output_tensor = np.zeros((1, total_seqlen, 2))
    parental = get_input_profile(np.array(short_profile), sub_seqlen)
    inputs, (reads, maf, maj_p, min_p) = sample_func(
        parental,
        [read_depth, read_depth],
        rd_scale_range,
        baf_scale_range,
        purity_range=[purity, purity],
    )
    input_tensor[:, :sub_seqlen, 0] = reads
    input_tensor[:, :sub_seqlen, 1] = maf
    output_tensor[:, :sub_seqlen, 0] = maj_p
    output_tensor[:, :sub_seqlen, 1] = min_p
    pos_info[:, :sub_seqlen, 0] = pos_vals(sub_seqlen)
    pos_info[:, :sub_seqlen, 1] = np.random.randint(1, 24)  # just one chrom
    return (
        torch.tensor(pos_info, dtype=int_dtype),
        torch.tensor(input_tensor, dtype=float_dtype),
        torch.tensor(output_tensor, dtype=float_dtype),
    )
