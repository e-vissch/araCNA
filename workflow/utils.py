import os

import pandas as pd


def get_bam_cases(config):
    info_ls = [
        (config[f"bam_loc_{i}"], config[f"case_file_{i}"])
        for i in config["case_include_indices"]
    ]

    df_ls = []

    def get_fpath(sample_line, sample_type, bam_loc):
        return f'{bam_loc}/{sample_line[f"id.{sample_type}"]}\
            /{sample_line[f"file_name.{sample_type}"]}'

    for bam_loc, case_file_name in info_ls:
        df = pd.read_csv(f"{config['data_dir']}/{case_file_name}", sep="\t")

        df = df.pivot(
            index="cases.0.case_id",
            columns="cases.0.samples.0.sample_type",
            values=["file_name", "id", "cases.0.samples.0.sample_id"],
        ).reset_index()
        df.columns = [
            ".".join(col).strip() if col[1] else col[0] for col in df.columns.values
        ]

        df = df.rename(
            columns={
                col: col.replace("Primary Tumor", "tumor").replace(
                    "cases.0.samples.0.sample_id", "sample_id"
                )
                for col in df.columns
            }
            | {"cases.0.case_id": "case_id"}
        )

        try:
            # rename normal
            for col_val in ["file_name", "sample_id", "id"]:
                df[f"{col_val}.normal"] = df[
                    f"{col_val}.Blood Derived Normal"
                ].combine_first(df[f"{col_val}.Solid Tissue Normal"])
        except KeyError:
            for col_val in ["file_name", "sample_id", "id"]:
                df[f"{col_val}.normal"] = df[f"{col_val}.Blood Derived Normal"]

        df["bam_normal"] = df.apply(
            get_fpath, sample_type="normal", bam_loc=bam_loc, axis=1
        )
        df["bam_tumor"] = df.apply(
            get_fpath, sample_type="tumor", bam_loc=bam_loc, axis=1
        )

        df["paths_exist"] = (df["bam_normal"].apply(os.path.exists)) & (
            df["bam_tumor"].apply(os.path.exists)
        )

        print(f"Missing paths for cases:\n{df[~df.paths_exist].case_id.values}")

        df = df[df.paths_exist]

        df_ls.append(df.set_index("case_id"))

    return pd.concat(df_ls)


def get_processed_cases(config):
    path = f"{config['data_dir']}/tcga_analysis/output/bams/"
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def get_bams(wildcards, cases, processed_cases=None, sample_type=False):
    if processed_cases and wildcards.case in processed_cases:
        return {}
    case_line = cases.loc[wildcards.case]
    if not sample_type:
        return {"bam_normal": case_line.bam_normal, "bam_tumor": case_line.bam_tumor}
    return {
        "bam": case_line.bam_normal
        if wildcards.sample_type == "normal"
        else case_line.bam_tumor
    }


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m{seconds:.3f}s"
