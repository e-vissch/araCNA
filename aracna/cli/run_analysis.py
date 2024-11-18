import click
from aracna.analysis.comparison_other_tools import (
    analyse_results_from_dir,
    get_joined_analyse_results,
    write_combined,
)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--base-dir", required=True)
@click.option("--model-key", required=True)
@click.option("--aracna-prefix", required=True)
@click.option("--ascat-prefix", required=False, default="TRUE")
def tcga_compare_from_separate(base_dir, model_key, aracna_prefix, ascat_prefix):
    get_joined_analyse_results(base_dir, model_key, aracna_prefix, ascat_prefix)


@cli.command()
@click.option("--base-dir", required=True)
@click.option("--model-key", required=True)
@click.option("--aracna-prefix", required=True)
@click.option("--out-prefix", required=False, default="")
@click.option("--ascat-prefix", required=False, default="TRUE")
def tcga_write_combined(base_dir, model_key, aracna_prefix, out_prefix, ascat_prefix):
    write_combined(base_dir, model_key, aracna_prefix, out_prefix, ascat_prefix)


@cli.command()
@click.option("--base-dir", required=True)
@click.option("--model-key", required=True)
@click.option("--include-plot", default=True)
@click.option("--out-prefix", required=False, default="")
def tcga_compare_from_combined(base_dir, model_key, include_plot, out_prefix):
    analyse_results_from_dir(base_dir, model_key, include_plot, out_prefix)


if __name__ == "__main__":
    cli()
