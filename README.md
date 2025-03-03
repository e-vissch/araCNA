# araCNA - a copy number alteration caller for somatic cells

## Installation

Required dependencies are listed in `requirements.txt` and this project assumes python 3.12 or above.

Installation time is less than 10 minutes. 

If you want install on a CPU/non-a100 gpu, do `pip install .`. Only Hyena models will run/train in this environment.

If you want to install on a100 GPU, then do `pip install .[a100_gpu]`. If you have issues with this, ensure your cudatoolkit, cuda and pytorch installations are aligned, and see [below](#troubleshooting-install) for troubleshooting. 

### Development

You can run either of these in editable mode (i.e using `pip install -e .`).

## Demo code- on simulated data
After installation
- You can run `araCNA_demo sim-infer` to run inference using the Hyena model. The output should be a plot. Note this samples a simulated dataset so if you run multiple times you can see the prediction on multiple samples of varying sampled complexity with length up to 650k. For most simulated samples, the model is very good. This should take less than a minute.
- `araCNA_demo small-train-cpu` will start training a new model on the CPU, you can terminate whenever you like.
- `araCNA_demo small-train-gpu` will start training a new model on the GPU, you can terminate whenever you like.

## Project overview

The project is structured as followed:

- `aracna`: the package that is installed and allows training/inference, it includes:
    - `src`: This is where all the deep learning code exists, where the model logic and architecture is defined.
    - `configs`: This is where all the hydra configs exist for training the deep learning models. It goes in tandem with the `src` directory.
    - `analysis`: This is where various scripts for performing analysis between methods, plotting etc exist.
    - `araCNA-models.zip`: zip files with pretrained models, installation should manage unzipping for you. The araCNA-models dir is also where any new models that you train with wandb will automatically be located.

- `workflow`: This is the workflows used to produce results for the manuscript from input BAMs. Please see below for more details on this. This is not included in the package distribution.
- `notebook_analysis`: closely linked to `analysis` and where final plots were produced interactively. This is not included in the package distribution.


## Note we edit some Mamba/Hyena definitions originally defined in:
- [mamba-ssm](https://github.com/state-spaces/mamba)- see `src/models/bi-mamba.py` for definitions/changes
- [hyena-dna](https://github.com/HazyResearch/hyena-dna)- most notably the `src/models/standalone_hyenadna.py`

## Running inference on tumour bam with already trained model
To do this, you'll need to provide a SNP allele loci csv file.
This is a file with headers: "chr", "position" "a0" "a1" with the reference and alternate alleles of common SNP loci. This file should be approximately 650k in length (if using with our pretrained models), as the models are trained up to this length, and will perform poorly if drastically different from this.

Please see [Reference files](#ref), if you would like to reproduce our loci file. 

Using this you can then run inference on your tumor bam file.
You can run the following command to do this all at once:

`araCNA_infer run-inference-on-bam --bam-file <link to bam file> --snp-file <link to ref snp subset file of ~650k length> --model-key <one of {pjflljt4,qwsvrrgk} if using pretrained (latter needs a100 gpu)> --out-dir <link to desired out dir> --num-processes <num>`

Will be much quicker if you have access to HPC with more processes (i.e 12-24). Note the bottleneck here is the preprocessing of the bam file to extract read depth and BAF data. 

Or else if you want to run multiple models on the same tumor file, you should only do the preprocessing once:

`araCNA_infer get-aracna-inputs --bam-file <link to bam file> --snp-file <link to ref snp subset file of ~650k length> --out-dir <link to desired out dir> --num-processes <num>`
`

Preprocessing step should output a csv file: "<outdir>/tumor_BAF_rd.txt".

And then can run inference with:

`araCNA_infer get-aracna-outputs --baf-rd-file <link to tumor_BAF_rd.txt> --model-key <one of {pjflljt4,qwsvrrgk} (latter needs a100 gpu)> --out-dir <link to desired out dir>`

Inference on preprocessed files should run within a few minutes- even on a CPU. The output files are:

- <output_dir>aracna_results_{model-key}.csv: A csv file containing each genomic location, the read depth, the BAF, and the major and minor copy numbers at each location. These can easily be processed to give segment CNs by looking at locations where minor/major CNs change.
- <output_dir>aracna_globals_{model-key}.csv: A csv containing the read depth (per copy number) estimation and the purity estimation. 

You can also do 

`araCNA_infer --help` or `araCNA_infer <command> --help` to see all the available options.

Where `araCNA_infer --help` will tell you available commands- `get-aracna-inputs` `run-inference-on-bam` and `get-aracna-outputs`.



## Reference files
Our input reference file is a combination of SNPs provided in the [ascatNGS](https://github.com/cancerit/ascatNgs/) repository (believe this should now be deprecated in favour of ascat which can also now do NGS), and their intersection with [ascat](https://github.com/VanLoo-lab/ascat) reference alleles that have been filtered for non-problematic loci. Both are generated using 1000 genomes, and basically have used different filters for selection- with the latter more stringent except for the former selecting alleles at least 1000 bp away (filter conditions [here](https://github.com/cancerit/ascatNgs/wiki/Human-reference-files-from-1000-genomes-VCFs)). We found this intersection to be a managable sequence length of ~650k, hence why we proceeded with it.

To reproduce the reference file creation, in your desired location, do:

```
mkdir snp_ref
cd snp_ref
```

Then download relevant reference ASCAT files, [reference](https://github.com/VanLoo-lab/ascat/tree/master/ReferenceFiles/WGS): 

`
mkdir G1000_alleles && wget -O G1000_alleles_WGS_hg38.zip https://zenodo.org/records/14008443/files/G1000_alleles_WGS_hg38.zip?download=1 && unzip G1000_alleles_WGS_hg38.zip -d G1000_alleles && rm G1000_alleles_WGS_hg38.zip
`

And relevant ascatNGS file:

`wget ftp://ftp.sanger.ac.uk/pub/cancer/dockstore/human/GRCh38_hla_decoy_ebv/CNV_SV_ref_GRCh38_hla_decoy_ebv_brass6+.tar.gz && tar -xzf CNV_SV_ref_GRCh38_hla_decoy_ebv_brass6+.tar.gz && mv CNV_SV_ref_GRCh38_hla_decoy_ebv_brass6+/ascat/SnpGcCorrections.tsv . && rm -rf CNV_SV_ref_GRCh38_hla_decoy_ebv_brass6* 
`

Then run:

`araCNA_infer get-snp-refs --ref-dir <your_dir_loc>/snp_ref` 

This will output a file `snp_allele_set.csv` to the `snp_ref` dir to be used in the inference calls.

The choice of SNPs is unlikely to matter, as long as they are prevalent enough in the population to capture enough heterozygous loci in the sample. For example likely any 650k subset of [ascat](https://github.com/VanLoo-lab/ascat) reference alleles could be used. Models can also be trained to a longer sequence length (we tested successfully up to 1million but did not test longer than this), and then run on a longer SNP set. The main sequence length limitation will be the memory limit of your GPU.



## Overview of src
The structure of this codebase is generalised to make it easier to try out different approaches.
This repository contains is the main source for the manuscript results, however some of the design has been implemented with further development in mind.

- DataModules
    - Where data logic belongs. At the moment we just have simulated datamodules but real datamodules could be easily added.
    - The existing simulated allows for custom samplers for different kinds of simulation.
    - The main sampler used in the manuscript is the `PurelySimulated` class

- Models
    - Where models belong - pytorch models that know nothing about lightning.

- Task Info
    - A task is kind of like a decoder head. We could have the entirely same base model, however, we might be able to represent the output in different ways- for example if we wanted an integer output then maybe we would have a task where the loss/decoder pair treats ouputs continuously and another that treats them as categories.
    - This logic is not defined on the decoder, because you might want the same decoder architecture for multiple tasks. The decoder class is a static attribute of the task though.
    - This logic is also decoupled from the datamodule, because you might want to train the same task on different data, or different tasks on the same data.
    - At the moment there is only really the `CatPaired` task, and the various derivatives of it (supervised, unsupervised) etc.


## Training a model

To train a model (using a GPU):

```
araCNA_train trainer.accelerator=gpu trainer.devices=1 ${any_more_hydra_args}
```

This project uses both [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) and [hydra](https://hydra.cc/docs/intro/), which give you a lot of things for free. It does mean sometimes having to adapt code to that framework though (if you are doing your own development).

Note if you forget the `trainer.accelerator=gpu trainer.devices=1` for a Mamba model it will not run.

If you want to use/edit the config yamls for training then you should install in editible mode. 


## Troubleshooting Install

Installing Mamba can be difficult as it requires having aligned cuda-toolkit, cuda and pytorch, which isn't guaranteed under pip or conda.

If you have issues with `pip install .[a100_gpu]`, try the following (ensuring a clean environment and that you are on an a100 GPU):

- `conda create --name aracna python=3.12`
- `conda activate aracna`
- `conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia`
- `conda install cuda-toolkit=12.4 -c nvidia`
- `pip install .[a100_gpu]`


## Overview of workflows
The folder workflows show the implementation of other methods used in the manuscript.
The `workflow/config.yaml` gives an indication of the required files for this analysis.
Most related files have been downloaded from each of the methods githubs/documentation.
Note, yaml doesn't actually work with environment variables/cross ref- any path beginning with "\$" has been replaced with a placeholder and should be replaced with true file locations.
Some of the workflow R scripts also point directly to file locations (also beginning with "\$"- (mainly $araCNA_dir, $data_dir and $read_mount, $write_mount for workflows using containers)).

For workflow to run you will need:

- R/R install with package installations for Battenberg, Ascat, HMM Copy and HMM Copy Utils
- Singularity images for CNV Kit
- Singularity image for ascatNGS, which is used as an input for battenberg to use that allele counter executable.

Workflows are run like `snakemake --workflow-profile workflow/snakemake_profile/ --snakefile  workflow/< file >.snk --jobs N
