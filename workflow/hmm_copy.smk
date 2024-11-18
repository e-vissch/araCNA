from workflow.utils import get_bam_cases, get_bams
import os

# snakemake --workflow-profile workflow/snakemake_profile/ --snakefile workflow/hmm_copy.smk --jobs N


configfile: "workflow/config.yaml"
workdir: "$araCNA_dir/"

cases = get_bam_cases(config)

window_size=500


rule all:
    input:
    #    preprocess=expand(f"{config['bam_output_dir']}/{{case}}/hmm_copy/{{sample_type}}_reads_{window_size}.wig", case=cases.index, sample_type=["tumor", "normal"]),
       for_hmmcopy=expand(f"{config['bam_output_dir']}/{{case}}/hmm_copy/out.snk_check", case=cases.index)


rule get_hmm_prelims:
    output:
        map_wig=f"{config["generic_dat_dir"]}/GRCh38.d1.vd1.fa.map.ws_{window_size}.wig",
        gc_wig=f"{config["generic_dat_dir"]}/GRCh38.d1.vd1.gc_{window_size}.wig"
    params:
        hmm_copy_dir=f"{config["modules_dir"]}/hmmcopy_utils-master",
        fasta_file=f"{config["generic_dat_dir"]}/GRCh38.d1.vd1.fa",
        fasta_readcount_size=75,
        window_size=window_size
    resources:
        mem="100GB"
    shell:
        """
        # {params.hmm_copy_dir}/util/mappability/generateMap.pl -b {params.fasta_file} -o {params.fasta_file}.map.bw

        # {params.hmm_copy_dir}/util/mappability/generateMap.pl {params.fasta_file} -w {params.fasta_readcount_size} -o {params.fasta_file}.map.bw

        {params.hmm_copy_dir}/bin/mapCounter -w {params.window_size} {params.fasta_file}.map.bw > {output.map_wig}

        {params.hmm_copy_dir}/bin/gcCounter {params.fasta_file} -w {params.window_size} > {output.gc_wig}
        """


def replace_func(file_name):
    return file_name.replace(".bam", ".bai")


rule hmmcopy_preprocess:
    input:
        unpack(partial(get_bams, cases=cases, sample_type=True))
    output:
        wig_out=f"{config['bam_output_dir']}/{{case}}/hmm_copy/{{sample_type}}_reads_{window_size}.wig"
    resources:
        cpus_per_task=2,
        mem="10GB"
    params:
        window=window_size,
        bam_link_dir=f"{config["interim_dat_dir"]}/input_copy/hmm_copy/",
        bam_symbolic_link=lambda wildcard, input: os.path.basename(input.bam),
        bai_orig=lambda wildcard, input: replace_func(input.bam),
        timing_out=f"{config['bam_output_dir']}/{{case}}/hmm_copy/{{sample_type}}_shell_timings.txt"
    shell:
        """
        # annoyingly read_ccounter expects file name to be .bam.bai in same loc
        mkdir -p {params.bam_link_dir}
        if [ ! -L {params.bam_link_dir}{params.bam_symbolic_link} ]; then
            ln -s {input.bam} {params.bam_link_dir}{params.bam_symbolic_link}
            echo "Symbolic link created."
        fi
        echo "processing_step\tpreprocess_{wildcards.sample_type}" > {params.timing_out}
        {{ time (
        readCounter -b {params.bam_link_dir}{params.bam_symbolic_link}
        readCounter -w {params.window} {params.bam_link_dir}{params.bam_symbolic_link} > {output.wig_out}
        ) ; }} 2>> {params.timing_out}
        """

rule hmm_process_bad_lines:
    input:
        wigs=[f"{config['bam_output_dir']}/{_case}/hmm_copy/{_sample_type}_reads_{window_size}.wig" for _case in cases.index for _sample_type in ['tumor', 'normal']] + [f"{config['generic_dat_dir']}/GRCh38.d1.vd1.gc_{window_size}.wig", f"{config['generic_dat_dir']}/GRCh38.d1.vd1.fa.map.ws_{window_size}.wig"]
    output:
        fix_wigs=[f"{config['bam_output_dir']}/{_case}/hmm_copy/{_sample_type}_reads_{window_size}_fix.wig" for _case in cases.index for _sample_type in ['tumor', 'normal']] + [f"{config['generic_dat_dir']}/GRCh38.d1.vd1.gc_{window_size}_fix.wig", f"{config['generic_dat_dir']}/GRCh38.d1.vd1.fa.map.ws_{window_size}_fix.wig"]
    params:
        script_file="workflow/workflow_scripts/rm_bad_hmm_lines.sh",
    shell:
        """
        # Iterate through all .wig files in the current directory
        echo "redo1"
        for input_file in {input.wigs}; do
            # Generate the output file name by replacing .wig with _fix.wig
            output_file=${{input_file%.wig}}_fix.wig

            # Call the call_func.sh script with the input and output files
            ./{params.script_file} $input_file $output_file
        done
        """


rule hmmcopy:
    input:
        tumor_wig=f"{config['bam_output_dir']}/{{case}}/hmm_copy/tumor_reads_{window_size}_fix.wig",
        normal_wig=f"{config['bam_output_dir']}/{{case}}/hmm_copy/normal_reads_{window_size}_fix.wig",
        gc_content=f"{config['generic_dat_dir']}/GRCh38.d1.vd1.gc_{window_size}_fix.wig",
        ref_mappable=f"{config['generic_dat_dir']}/GRCh38.d1.vd1.fa.map.ws_{window_size}_fix.wig",
    output:
        snk_check=f"{config['bam_output_dir']}/{{case}}/hmm_copy/out.snk_check"
    params:
        scripts_dir="workflow/workflow_scripts",
        out_stub=f"{config['bam_output_dir']}/{{case}}/hmm_copy/hmm_copy",
        timing_out=f"{config['bam_output_dir']}/{{case}}/hmm_copy/shell_timings.txt"
    resources:
        mem="50GB"
    shell:
        """
        module load R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2
        echo "processing_step\tmain_hmm_copy" > {params.timing_out}
        {{ time ( Rscript {params.scripts_dir}/hmm_copy.R --normal_reads {input.normal_wig} --tumor_reads {input.tumor_wig} --gc_content {input.gc_content} --ref_mappable {input.ref_mappable} -o {params.out_stub} ) ; }} 2>> {params.timing_out}
        touch {output.snk_check}
        """
