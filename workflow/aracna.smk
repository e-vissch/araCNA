import pandas as pd
from workflow.utils import get_bam_cases, get_bams, get_processed_cases
from aracna.analysis.tcga_data import write_case_csvs, write_cases_csvs_from_lists
from aracna.analysis.comparison_other_tools import write_combined, analyse_results_from_dir

configfile: "workflow/config.yaml"
workdir: "$araCNA_dir/"


def get_processed_cases_simple(config):
    path  = f"{config["data_dir"]}/tcga_analysis/output/bams/"
    return [d for d in os.listdir(path) if os.path.exists(os.path.join(path, d,'aracna/tumor_BAF_rd.txt'))]

processed_cases = get_processed_cases_simple(config)

model_keys = sorted(['pjflljt4','qwsvrrgk'])

model_out_str = '_'.join(model_keys)

rule all:
    input:
        aracna_cnas= expand(f"{config['bam_output_dir']}/{{case}}/aracna/{{file_prefix}}tumor_aracna_results_{{model_key}}.csv", case=processed_cases, model_key=model_keys, file_prefix=['normal_depth_']),
        joined_processed= expand(f"{config['bam_output_dir']}/{{case}}/joined_results/{{file_prefix}}summary_stats_{model_out_str}.json", case=processed_cases, file_prefix=['normal_depth_']),


#  usually faster to just run on one GPU node sequentially than wait for multiple.
# also faster to run sequentially internally rather that sequentially external to python program
rule get_aracna_bulk:
    input:
        aracna_inputs=expand(f"{config['bam_output_dir']}/{{_case}}/aracna/tumor_BAF_rd.txt", _case=processed_cases)
    output:
        aracna_cnas=[f"{config['bam_output_dir']}/{case}/aracna/{{file_prefix}}tumor_aracna_results_{{model_key}}.csv" for case in processed_cases],
        aracna_globs=[f"{config['bam_output_dir']}/{case}/aracna/{{file_prefix}}tumor_aracna_globals_{{model_key}}.csv" for case in processed_cases],
    params:
        out_stubs=[f"{config['bam_output_dir']}/{case}/aracna/{{file_prefix}}tumor_" for case in processed_cases],
        depth_type="depth",
        detailed=True
    resources:
        slurm_partition="gpu_short",
        constraint='a100', # note this and above only necessary for mamba implementations
        mem='50G',
        slurm_extra="--gpus 1"
    run:
        write_cases_csvs_from_lists(params.out_stubs, wildcards.model_key, input.aracna_inputs, detailed=params.detailed)



rule get_joined:
    input:
        infiles=[f"{config['bam_output_dir']}/{{case}}/aracna/{{file_prefix}}tumor_aracna_results_{model_key}.csv" for model_key in model_keys],
    output:
        joined_file=f"{config['bam_output_dir']}/{{case}}/joined_results/{{file_prefix}}sequence_{model_out_str}.csv",
        glob_file=f"{config['bam_output_dir']}/{{case}}/joined_results/{{file_prefix}}globals_{model_out_str}.csv",
    params:
        base_dir=f"{config['bam_output_dir']}/{{case}}",
        aracna_prefix=f"{{file_prefix}}tumor_aracna_"
    resources:
        mem="50G",
        slurm_partition="short",
    run:
        write_combined(params.base_dir, model_keys, params.aracna_prefix, wildcards.file_prefix)


rule do_analysis:
    input:
        joined_file=f"{config['bam_output_dir']}/{{case}}/joined_results/{{file_prefix}}sequence_{model_out_str}.csv",
        glob_file=f"{config['bam_output_dir']}/{{case}}/joined_results/{{file_prefix}}globals_{model_out_str}.csv",
    output:
        joined_file=f"{config['bam_output_dir']}/{{case}}/joined_results/{{file_prefix}}summary_stats_{model_out_str}.json",
    params:
        base_dir=f"{config['bam_output_dir']}/{{case}}",
        include_plot=False
    resources:
        mem="80G",
        slurm_partition="short",
    run:
        analyse_results_from_dir(params.base_dir, model_keys, params.include_plot, file_prefix=wildcards.file_prefix)



# to run:
# snakemake --workflow-profile workflow/snakemake_profile/ --snakefile workflow/aracna_new.smk --cores 6
# avoids group rule taking more resources than necessary
