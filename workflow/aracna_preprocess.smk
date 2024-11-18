import pandas as pd
from workflow.utils import get_bam_cases, get_bams
from aracna.analysis.process_bams import write_full_df, get_file_info

configfile: "workflow/config.yaml"
workdir: "$araCNA_dir/"

cases = get_bam_cases(config)

rule all:
    input:
        aracna_inputs = expand(f"{config['bam_output_dir']}/{{case}}/aracna/tumor_BAF_rd.txt", case=cases.index),

rule write_snp_info:
    input:
        loci_file=config['loci_file'],
        snp_alleles=expand(f"{config['allele_base_dir']}chr{{chr_val}}.txt", chr_val=list(range(1, 23)) + ["X"]),
    params:
        snp_allele_dir=config['allele_base_dir'],
    output:
        out_file=config['combined_loci_allele_file'],
    run:
        write_full_df(input.loci_file, params.snp_allele_dir, output.out_file)


def replace_func(file_name):
    return file_name.replace(".bam", ".bai")

rule process_bams:
    # Only have read privileges to the bam files, so create symbolic link and index.
    input:
        unpack(partial(get_bams, cases=cases))
    output:
        bam_symbolic_link=f"{config["interim_dat_dir"]}/input_copy/{{case}}_tumor.bam",
        bai_file=f"{config["interim_dat_dir"]}/input_copy/{{case}}_tumor.bam.bai"
    params:
        bai_original=lambda wildcard, input: replace_func(input.bam_tumor)
    shell:
        """
        module load R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2
        if [ ! -L {output.bam_symbolic_link} ]; then
            ln -s {input.bam_tumor} {output.bam_symbolic_link}
            echo "Symbolic link created."
        fi

        if [ ! -e {output.bai_file} ]; then # unsure if neccessary
            if [ -e {params.bai_original} ]; then
                ln -s {params.bai_original} {output.bai_file}
                echo "Symbolic link created."
            elif [ -e {input.bam_tumor}.bai ]; then
                ln -s {input.bam_tumor}.bai {output.bai_file}
                echo "Symbolic link created."
            else
                samtools index {output.bam_symbolic_link}
            fi
        fi
        """

rule get_baf_rd:
    input:
        bam_symbolic_link=f"{config["interim_dat_dir"]}/input_copy/{{case}}_tumor.bam",
        snp_file=config['combined_loci_allele_file'],
    output:
        aracna_input=f"{config['bam_output_dir']}/{{case}}/aracna/tumor_BAF_rd.txt"
    params:
        num_processes=lambda wildcards, resources: resources.cpus_per_task * 2
    resources:
        cpus_per_task=24,
        mem="50GB"
    run:
        get_file_info(input.bam_symbolic_link, input.snp_file, output.aracna_input, params.num_processes)
