import os
import pandas as pd

configfile: "workflow/config.yaml"
workdir: "$araCNA_dir/"
# snakemake --workflow-profile workflow/snakemake_profile/ --snakefile workflow/ascat_cel.smk --jobs N

def get_samples(wildcards):
      checkpoint_dir = checkpoints.preprocess_cels.get(**wildcards).output[0]
      return expand(f"{config['cel_output_dir']}/ascat/{{sample}}.{{file}}.txt", sample = glob_wildcards(os.path.join(checkpoint_dir, "sample.{i}")).i, file=["segments", "acf", "ploidy"])


matched_df = pd.read_csv(f"{config["cel_output_dir"]}/matched_cels.tsv", sep="\t").set_index('case_id')


rule all:
    input:
        # get_samples
        expand(f"{config['cel_output_dir']}/ascat/{{case}}.segments.txt", case=matched_df.index.tolist())



rule preprocess_cels:
    # could edit now to remove CELs that have been dropped
    input:
        # cel_file=expand(cel_dir[{cel}], cel=list(cel_files.keys()))
        cel_files= f"{config['cel_output_dir']}/CELfiles.txt"
    output:
        out_files = expand(f"{config['cel_output_dir']}/samples/sample.{{sample}}", sample=[c.removesuffix(".CEL") for c in matched_df["sample_name.normal"].tolist() + matched_df["sample_name.tumor"].tolist()])
        # out_file=f"{config['cel_output_dir']}/out_lrr_baf.txt"
        # directory(f"{config['cel_output_dir']}/samples")
    params:
        apt_dir=f"{config['apt_data_dir']}",
        out_dir=f"{config['cel_output_dir']}",
        affy_dir=f"{config['penn_cnv_affy_dir']}"
    resources:
        cpus_per_task=24,
        mem="100GB"
    shell:
        """
        apt-probeset-genotype -c {params.apt_dir}/GenomeWideSNP_6.cdf -a birdseed --read-models-birdseed {params.apt_dir}/GenomeWideSNP_6.birdseed.models --special-snps {params.apt_dir}/GenomeWideSNP_6.specialSNPs --out-dir {params.out_dir}/preprocess --cel-files {input.cel_files}
        # step 1.2:
        apt-probeset-summarize --cdf-file {params.apt_dir}/GenomeWideSNP_6.cdf --analysis quant-norm.sketch=50000,pm-only,med-polish,expr.genotype=true --target-sketch {params.affy_dir}/hapmap.quant-norm.normalization-target.txt --out-dir {params.out_dir}/preprocess --cel-files {input.cel_files}
        # step 1.4:
        normalize_affy_geno_cluster.pl {params.affy_dir}/hapmap.genocluster {params.out_dir}/preprocess/quant-norm.pm-only.med-polish.expr.summary.txt -locfile {params.affy_dir}/affygw6.hg38.pfb -out {output.out_file}
        kcolumn.pl {output.out_file} split 2 -tab -head 3 -name -out {params.out_dir}/samples/sample
        """

def get_lrr_baf(wildcards):
    case_line = matched_df.loc[wildcards.case]
    get_file = lambda val: f"{config['cel_output_dir']}/samples/sample.{val.removesuffix(".CEL")}"
    return {"normal_llr_baf": get_file(case_line["sample_name.normal"]), "tumor_llr_baf": get_file(case_line["sample_name.tumor"])}



rule run_ascat_cel:
    input:
        unpack(get_lrr_baf)
    output:
        segments=f"{config['cel_output_dir']}/ascat/{{case}}.segments.txt",
        # purity=f"{config['cel_output_dir']}/ascat/{{sample}}.acf.txt",
        # ploify=f"{config['cel_output_dir']}/ascat/{{sample}}/ploidy.txt",
    params:
        scripts_dir="workflow/ascat_scripts",
        data_dir=f"{config['cel_output_dir']}/samples",
        snp_gc_dir=f"{config['generic_dat_dir']}/SNP6_info",
        birdseed_dir=f"{config['cel_output_dir']}/preprocess",
        out_dir=f"{config['cel_output_dir']}/ascat",
        snp_pos_file=f"{config['generic_dat_dir']}/SNP6_info/SNP6_remapped_hg38.txt",
        case=f"{{case}}",
    resources:
        cpus_per_task=1,
        mem="10GB"
    shell:
        """
        module load R-bundle-Bioconductor/3.14-foss-2021b-R-4.1.2
        Rscript {params.scripts_dir}/ascat_cel.R -c {params.case} -t {input.tumor_llr_baf} -n {input.normal_llr_baf} --snp_pos_file {params.snp_pos_file} --snp_gc_dir {params.snp_gc_dir} --birdseed_dir {params.birdseed_dir}  --out_dir {params.out_dir}
        """
