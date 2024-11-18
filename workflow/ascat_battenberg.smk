# To run:
# cd $main_dir/projects/araCNA/
# bash run_scripts/env.sh
# snakemake --workflow-profile workflow/snakemake_profile/ --jobs N

import pandas as pd
from workflow.utils import get_bam_cases, get_bams

configfile: "workflow/config.yaml"
workdir: "$araCNA_dir/"


cases = get_bam_cases(config)


rule all:
    input:
       for_ascat_wgs=expand(f"{config['bam_output_dir']}/{{case}}/ascat_wgs/corr_{{corr}}/ascat.snk_check", case=cases.index, corr=["TRUE"]),
       for_battenberg=expand(f"{config['bam_output_dir']}/{{case}}/battenberg/bat.snk_check", case=cases.index)


def get_names(wildcards):
    case_line = cases.loc[wildcards.case]
    return {"normal": case_line["sample_id.normal"], "tumor": case_line["sample_id.tumor"]}


rule fix_vcfs:
    input:
        vcf=f"{config['generic_dat_dir']}/battenberg_hg38/beagle/chr{{chrom}}.1kg.phase3.v5a_GRCh38nounref.vcf.gz",
    output:
        vcf_fixed=f"{config['generic_dat_dir']}/battenberg_hg38/beagle/fixed_vcfs/chr{{chrom}}.1kg.phase3.v5a_GRCh38nounref.vcf.gz",
    params:
        workdir=f"{config['generic_dat_dir']}/battenberg_hg38/beagle/",
        interim_vcf="chr{chrom}.1kg.phase3.v5a_GRCh38nounref.vcf",
    shell:
        """
        module load BCFtools/1.14-GCC-11.2.0
        cd {params.workdir}
        bcftools view {input.vcf} -o fixed_vcfs/{params.interim_vcf}
        awk '{{if($0 !~ /^#/) print "chr"$0; else print $0}}' fixed_vcfs/{params.interim_vcf} > fixed_vcfs/fix_{params.interim_vcf}
        bcftools view -Oz fixed_vcfs/fix_{params.interim_vcf} -o {output.vcf_fixed}
        rm fixed_vcfs/fix_{params.interim_vcf}
        rm fixed_vcfs/{params.interim_vcf}
        """

def replace_func(file_name):
    return file_name.replace(".bam", ".bai")

rule process_bams:
    input:
        unpack(partial(get_bams, cases=cases, sample_type=True))
    output:
        bam_symbolic_link=f"{config["interim_dat_dir"]}/input_copy/{{case}}_{{sample_type}}.bam",
        bai_file=f"{config["interim_dat_dir"]}/input_copy/{{case}}_{{sample_type}}.bam.bai"
    params:
        bai_original=lambda wildcard, input: replace_func(input.bam)
    shell:
        """
        module load R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2
        if [ ! -L {output.bam_symbolic_link} ]; then
            ln -s {input.bam} {output.bam_symbolic_link}
            echo "Symbolic link created."
        fi

        if [ ! -e {output.bai_file} ]; then # unsure if neccessary
            if [ -e {params.bai_original} ]; then
                ln -s {params.bai_original} {output.bai_file}
                echo "Symbolic link created."
            elif [ -e {input.bam}.bai ]; then
                ln -s {input.bam}.bai {output.bai_file}
                echo "Symbolic link created."
            else
                samtools index {output.bam_symbolic_link}
            fi
        fi
        """


rule run_battenberg:
    input:
        bam_normal=f"{config["interim_dat_dir"]}/input_copy/{{case}}_normal.bam",
        bam_tumor=f"{config["interim_dat_dir"]}/input_copy/{{case}}_tumor.bam",
        vcfs_fixed=expand(f"{config['generic_dat_dir']}/battenberg_hg38/beagle/fixed_vcfs/chr{{chrom}}.1kg.phase3.v5a_GRCh38nounref.vcf.gz", chrom=list(range(1, 24)) + ["X"])
    output:
        snk_check=f"{config['bam_output_dir']}/{{case}}/battenberg/bat.snk_check"
    params:
        names=get_names,
        base_dir=f"{config['generic_dat_dir']}/battenberg_hg38",
        out_dir=f"{config['bam_output_dir']}/{{case}}/battenberg",
        affy_dir=f"{config['penn_cnv_affy_dir']}",
        scripts_dir="workflow/workflow_scripts",
        allele_exec=f"{config['aracna_loc']}/workflow/workflow_scripts/allele_counter.sh",
        timing_out=f"{config['bam_output_dir']}/{{case}}/battenberg/shell_timings.txt"
    resources:
        cpus_per_task=12,
        mem="300GB"
    shell:
        """
        module load R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2
        echo "processing_step\tbattenberg_complete" > {params.timing_out}
        {{ time ( Rscript {params.scripts_dir}/battenberg_wgs.R --basedir {params.base_dir} -t {params.names[tumor]} -n {params.names[normal]} --tb {input.bam_tumor} --nb {input.bam_normal} -o {params.out_dir} --sex Female --alleleCounter_exe {params.allele_exec} --cpu {resources.cpus_per_task} ) ; }} 2> {params.timing_out}
        touch {output.snk_check}
        """




rule run_ascat_wgs:
    input:
        bam_normal=f"{config["interim_dat_dir"]}/input_copy/{{case}}_normal.bam",
        bam_tumor=f"{config["interim_dat_dir"]}/input_copy/{{case}}_tumor.bam",
    output:
        snk_check=f"{config['bam_output_dir']}/{{case}}/ascat_wgs/corr_{{corr}}/ascat.snk_check"
    params:
        names=get_names,
        ascat_data_dir=f"{config['generic_dat_dir']}/ascat_wgs",
        out_dir=f"{config['bam_output_dir']}/{{case}}/ascat_wgs/corr_{{corr}}",
        scripts_dir="workflow/workflow_scripts",
        allele_exec=f"{config['aracna_loc']}/workflow/workflow_scripts/allele_counter.sh",
        nthreads=lambda wildcards, resources: resources.cpus_per_task * 2,
        timing_out=f"{config['bam_output_dir']}/{{case}}/ascat_wgs/corr_{{corr}}/shell_timings.txt"
    resources:
        cpus_per_task=12,
        mem="50GB"
    shell:
        """
        module load R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2
        echo "processing_step\tascat_complete" > {params.timing_out}
        {{ time ( Rscript {params.scripts_dir}/ascat_wgs.R -d {params.ascat_data_dir} -t {params.names[tumor]} -n {params.names[normal]} --tumor_bam {input.bam_tumor} --normal_bam {input.bam_normal} -o {params.out_dir} --alleleCounter_exe {params.allele_exec} --nthreads {params.nthreads} --include_correction {wildcards.corr} ) ; }} 2> {params.timing_out}
        touch {output.snk_check}
        """
