import sys
import click

from aracna.cli.dev_entrypoint import small_run, small_run_gpu

from aracna.analysis.sim_inference_utils import get_simulated_infer


@click.group()
def cli():
    pass


@cli.command()
@click.argument('args', nargs=-1)
def small_train_cpu(args):
    # kinda gross but hydra just looks at CLI
    del sys.argv[1]
    sys.argv.append('experiment=simple_start')
    small_run()


@cli.command()
@click.argument('args', nargs=-1)
def small_train_gpu(args):
    # kinda gross but hydra just looks at CLI
    del sys.argv[1]
    sys.argv.append('experiment=simple_start')    
    small_run_gpu()


@cli.command(help="Run inference on simulated data. Ensure that on a100-GPU if running a Mamba model.")
@click.option("--model-key", required=False, default="pjflljt4")
@click.option("--read-depth", required=False, default=15)
@click.option("--purity", required=False, default=1)
def sim_infer(model_key, read_depth, purity):
    get_simulated_infer(model_key, read_depth, purity)



if __name__ == "__main__":
    cli()