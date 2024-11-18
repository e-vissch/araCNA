# araCNA - a copy number alteration caller for somatic cells


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


## Installation

Unfortunately due to pytorch and CUDA interplay, `setup.py` cannot manage everything directly.
We recommend you read all the installation instructions before running any commands, so you know which option to choose, you may run into issues on the GPU if you do not install in the correct order.

Required dependencies are listed in `requirements.txt` and this project assumes python 3.12 or above.

Installation time is 10 minutes. 


### Setup without GPU

You can still use hyena without a GPU (although training will be slow). But the already trained model can be used for inference on a CPU.

For this you can first create environment (e.g using conda):

- `conda create --name aracna python=3.12`
- `conda activate aracna`

And then inside your environment:

`pip install .`

This is also nice for local dev on a CPU (though in this case you'll want to install in editable mode `pip install -e .`)

### Setup on any GPU

If you want to train a Hyena model, or run the Mamba pretrained model this is what you should do (noting you must be on a computer with a GPU).

If you have a version of CUDA already installed, then install torch with that version, run `nvidia-smi` or `nvcc --version`. If this shows a driver of `12.6` it means that it can support CUDA up until that version.

If you have a conda you can install different CUDA software versions, or else if you are on the HPC- they might have prepackaged CUDA/pytorch modules available.


#### If no available CUDA versions, you need to use conda which will handle cuda install for you.
You can do something like the following:

`conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia`


#### Using existing CUDA verison

Inside your environment run:

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu<version>`

Where `<version>` might be `124` for CUDA 12.4. Note this URL might change in future, so worth checking the pytorch website. Note also, only certain CUDA versions come directly compiled for pytorch, if you want to use a different CUDA version with pytorch then you will have to compile directly from source.

What might be easier is installing as above with conda, that will update the CUDA version in your local environment to one compatible with pytorch.

#### Once compatible pytorch + CUDA installed

You can simply run `pip install .`. However, if you want to run with Mamba, on a100 gpu- see below.


### Setup with a100 GPU

If you want to train/infer using a Mamba model, then you'll need to be on an a100 gpu. For one the Mamba dependencies to install, you will also need cuda-toolkit. If you have a conda/cuda module on your HPC you can first try loading that then running `pip install .['a100_gpu']`, if that doesn't work then please follow the below instructions.

Unfortunately, installing cuda-toolkit can be a pain due to glitchy dependency management, if you do not ensure compatible versions between cuda/pytorch/cuda-toolkit.
The following has been tested although might fail in future:

- `conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia`
- `conda install cuda-toolkit=12.4 -c nvidia`
- `pip install .[a100_gpu]`

If you have problems installing have a look at [blog post](https://www.blopig.com/blog/2024/01/tip-and-tricks-to-correct-a-cuda-toolkit-installation-in-conda/) for tips to fix.

Once you do this once, araCNA using Hyena should work on any HPC architecture (CPU/GPU), and Mamba will work on the a100 GPU.


### Development

You can run either of these in editable mode (i.e using `pip install -e .`) but note that in this case the araCNA-models directory may need to be manually unzipped.

## Demo code- on simulated data

If you would like to quickly demo the code, without installing on the GPU, please follow CPU install instructions.
- You can then run `araCNA_demo sim-infer` to run inference using the Hyena model. The output should be a plot. Note this samples a simulated dataset so if you run multiple times you can see the prediction on multiple samples of varying sampled complexity with length up to 650k. For most simulated samples, the model is very good.
- `araCNA_demo small-train-cpu` will start training a new model on the CPU.
- `araCNA_demo small-train-gpu` will start training a new model on the GPU.



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
