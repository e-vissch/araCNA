import os

import click
from aracna.analysis.process_bams import get_file_info, write_full_df
from aracna.analysis.tcga_data import write_case_csvs


@click.group()
def cli():
    pass


@cli.command()
@click.option("--ref-dir", required=True)
def get_snp_refs(ref_dir):
    if not os.path.exists(f"{ref_dir}/G1000_alleles") or not os.path.exists(
        f"{ref_dir}/SnpGcCorrections.tsv"
    ):
        print("Missing reference files, please see ReadMe for details")
    write_full_df(
        f"{ref_dir}/SnpGcCorrections.tsv",
        f"{ref_dir}/G1000_alleles/G1000_alleles_hg38_",
        f"{ref_dir}/snp_allele_set.csv",
    )


@cli.command()
@click.option("--bam-file", required=True)
@click.option("--snp-file", required=True)
@click.option("--out-dir", required=True)
@click.option("--file-prefix", default="")
@click.option("--num-processes", required=False, default=2)
def get_aracna_inputs(bam_file, snp_file, out_dir, file_prefix, num_processes):
    out_stub = f"{out_dir}/{file_prefix}"
    baf_rd_file = f"{out_stub}tumor_BAF_rd.txt"
    get_file_info(bam_file, snp_file, baf_rd_file, num_processes)


@cli.command()
@click.option("--bam-file", required=True)
@click.option("--snp-file", required=True)
@click.option("--model-key", required=True)
@click.option("--out-dir", required=True)
@click.option("--file-prefix", default="")
@click.option("--num-processes", required=False, default=2)
@click.option("--detailed", default=False)
@click.option("--include-plot", default=False)
def run_inference_on_bam(
    bam_file, snp_file, model_key, out_dir, file_prefix, num_processes, detailed, include_plot
):
    out_stub = f"{out_dir}/{file_prefix}"
    baf_rd_file = f"{out_stub}tumor_BAF_rd.txt"
    get_file_info(bam_file, snp_file, baf_rd_file, num_processes)
    write_case_csvs(
        out_stub, model_key, input_file=baf_rd_file, detailed=detailed, include_plot=include_plot
    )


@cli.command()
@click.option("--baf-rd-file", required=True)
@click.option("--model-key", required=True)
@click.option("--out-dir", required=True)
@click.option("--file-prefix", default="")
@click.option("--detailed", default=False)
@click.option("--include-plot", default=False)
def get_aracna_outputs(baf_rd_file, model_key, out_dir, file_prefix, detailed, include_plot):
    out_stub = f"{out_dir}/{file_prefix}"
    write_case_csvs(
        out_stub, model_key, input_file=baf_rd_file, detailed=detailed, include_plot=include_plot
    )


if __name__ == "__main__":
    cli()
