# REF FLAT: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refFlat.txt.gz
from workflow.utils import get_bam_cases, get_bams
import os, glob
import pandas as pd

configfile: "workflow/config.yaml"
workdir: "$araCNA_dir/"

cases = get_bam_cases(config)

def get_processed_cases():
    path  = f"{config["data_dir"]}/tcga_analysis/output/bams/"
    return {d for d in os.listdir(path) if os.path.exists(os.path.join(path, d,'cnv_kit/snk.check'))}


def get_cnv_kit_output(wildcards, cases):
    seg_files = glob.glob(os.path.join(f"{config['interim_dat_dir']}/output/bams/{wildcards.case}/cnv_kit/", "*.cns"))
    for f in seg_files:
        if not ".bintest.cns" in f and not ".call.cns" in f:
            return {"seg_file": f}


def get_purity(wildcards):
    if wildcards.purity_suffix == '_pure':
        return ''
    elif wildcards.purity_suffix == '_ascat_purity':
        ascat_dir=f"{config['bam_output_dir']}/{wildcards.case}/ascat_wgs/corr_TRUEpenalty_70"
        ascat_globals = pd.read_csv(f"{ascat_dir}/qc.csv")
        return f'--purity {ascat_globals.purity.item()}'



def get_ploidy(wildcards):
    if wildcards.purity_suffix == '_pure':
        return ''
    elif wildcards.purity_suffix == '_ascat_purity':
        ascat_dir=f"{config['bam_output_dir']}/{wildcards.case}/ascat_wgs/corr_TRUEpenalty_70"
        ascat_globals = pd.read_csv(f"{ascat_dir}/qc.csv")
        return f'--ploidy {round(ascat_globals.ploidy.item())}'


# other_case_names = get_processed_cases()
other_case_names = []

rule all:
    input:
        joined_processed = expand(f"{config['bam_output_dir']}/{{case}}/cnv_kit/tumor{{purity_suffix}}.cns.baf_call", case=get_processed_cases(), purity_suffix=['_pure', '_ascat_purity']),



rule get_bed:
    input:
        snp_ref=f"{config['combined_loci_allele_file']}" # this is an output of aracna_preprocess- not nec. just need some ref snp_bed as input to get_vcfs otherwise next step v slow (includes all SNVs)
    output:
        snp_bed=f"{config['generic_dat_dir']}/snp_ref/snp_allele_set.bed"
    shell:
    """
    tail -n +2 {input.snp_ref} | awk -F, '
        {
            chr = $1 == "23" ? "chrX" : "chr" $1
            start_pos = $2 - 1
            end_pos = $2
            print chr "\t" start_pos "\t" end_pos
        }' > {output.snp_bed}
    """he



rule get_vcfs:
    input:
        unpack(partial(get_bams, cases=cases)),
        snp_bed=f"{config['generic_dat_dir']}/snp_ref/snp_allele_set.bed" # not nec. but otherwise step very slow as includes somatic snvs
    output:
        pileup=f"{config['interim_dat_dir']}/output/bams/{{case}}/cnv_kit/tumor.pileup",
        vcf=f"{config['interim_dat_dir']}/output/bams/{{case}}/cnv_kit/allele_calls.vcf"
    params:
        generic_dat_dir=config['generic_dat_dir'],
        num_processes=lambda wildcards, resources: resources.cpus_per_task,
        modules_dir=config["modules_dir"]
    resources:
        mem="100G",
        slurm_partition="short",
        cpus_per_task=12,
    shell:
        """
        module load R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2

        samtools mpileup -f {params.generic_dat_dir}/GRCh38.d1.vd1.fa {input.bam_tumor} --positions {input.snp_bed} > {output.pileup}

        java -jar {params.modules_dir}/varscan-2.4.6/VarScan.v2.4.6.jar mpileup2snp {output.pileup} --output-vcf 1 > {output.vcf}
        """



rule run_cnvkit:
    input:
        unpack(partial(get_bams, cases=cases))
    output:
        snk_check=f"{config['interim_dat_dir']}/output/bams/{{case}}/cnv_kit/snk.check"
    resources:
        mem="100G",
        slurm_partition="short",
        cpus_per_task=12,
    params:
        num_processes=lambda wildcards, resources: resources.cpus_per_task,
        generic_dat_dir=config['generic_dat_dir'],
        bam_locs=config['bam_locs'],
        interim_dat_dir=config['interim_dat_dir'],
        image_loc=config['cnv_kit_imag_loc']
    shell:
        """
        mkdir -p {params.interim_dat_dir}/output/bams/{wildcards.case}/cnv_kit
        singularity exec \
        --bind {params.generic_dat_dir}:{params.generic_dat_dir}:rw \
        --bind {params.bam_locs}:{params.bam_locs}:ro \
        --bind {params.interim_dat_dir}:{params.interim_dat_dir}:rw \
        {params.image_loc} \
        /bin/bash -c "cd {params.generic_dat_dir}/cnv_kit/ && /opt/conda/bin/cnvkit.py batch {input.bam_tumor} -n {input.bam_normal} -m wgs -f {params.generic_dat_dir}/GRCh38.d1.vd1.fa --annotate {params.generic_dat_dir}/refFlat.txt -p {params.num_processes} -d {params.interim_dat_dir}/output/bams/{wildcards.case}/cnv_kit/"
        touch {output.snk_check}
        """


rule run_cnvkit_call_cns:
    input:
        vcf=f"{config['interim_dat_dir']}/output/bams/{{case}}/cnv_kit/allele_calls.vcf",
        snk_check=f"{config['interim_dat_dir']}/output/bams/{{case}}/cnv_kit/snk.check"
    output:
        call_file = f"{config['bam_output_dir']}/{{case}}/cnv_kit/tumor{{purity_suffix}}.cns.baf_call"
    resources:
        mem="50G",
        slurm_partition="short",
        cpus_per_task=6,
    params:
        workdir= f"{config['bam_output_dir']}/{{case}}/cnv_kit/",
        seg_file=lambda wildcards: get_cnv_kit_output(wildcards, cases=cases)["seg_file"],
        num_processes=lambda wildcards, resources: resources.cpus_per_task,
        generic_dat_dir=config['generic_dat_dir'],
        interim_dat_dir=config['interim_dat_dir'],
        image_loc=config['cnv_kit_imag_loc'],
        purity=get_purity,
        ploidy=get_ploidy,
    group: "subgroup2"
    shell:
        """
        mkdir -p {params.interim_dat_dir}/output/bams/{wildcards.case}/cnv_kit
        echo {params.purity}
        singularity exec --pwd / \
        --bind {params.interim_dat_dir}:{params.interim_dat_dir}:rw \
        {params.image_loc} \
        /bin/bash -c "cd {params.workdir} && /opt/conda/bin/cnvkit.py call {params.seg_file} {params.purity} {params.ploidy} -v {input.vcf} -o {output.call_file}"
        """

#snakemake --workflow-profile workflow/snakemake_profile --snakefile workflow/cnv_kit.smk --cores 12 --jobs 100 --resources mem_mb=50000
