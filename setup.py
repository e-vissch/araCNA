import os
import zipfile

from setuptools import find_packages, setup
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Custom install command to unzip the single reference data zip file after installation."""

    def run(self):
        # Run the standard install command
        install.run(self)

        # If file doesn't exist in source, try installed package path
        if not os.path.exists(zip_file_path := os.path.abspath(
            "aracna/araCNA-models.zip")):
            return

        # Define target directory
        target_dir = os.path.dirname(zip_file_path)

        # Unzip the file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
            print(f"Unzipped {zip_file_path} to {target_dir}")


# Function to read requirements.txt
def parse_requirements(filename):
    with open(filename) as file:
        return file.read().splitlines()


setup(
    name="araCNA",
    version="0.1.0",
    description="Project package to go with araCNA manuscript.",
    author="Ellen Visscher",
    packages=['aracna'],
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "a100_gpu": ["causal-conv1d>=1.1.0", "mamba-ssm"],  # Specify GPU dependencies
    },
    python_requires=">=3.12",  # Specify minimum Python version
    package_data={"aracna": ["**/*"]},
    include_package_data=True,
    setup_requires=['setuptools_scm'],
    entry_points={
        "console_scripts": [
            "araCNA_train=aracna.cli.train:main",
            "araCNA_infer=aracna.cli.infer_bams:cli",
            "araCNA_demo=aracna.cli.demo:cli",
        ],
    },
    cmdclass={
        "install": CustomInstallCommand,
    },
)
