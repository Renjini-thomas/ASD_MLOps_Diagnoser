import subprocess
import yaml
from pathlib import Path
from src.utils.logger import logger


class DataIngestion:

    def __init__(self):
        config_path = Path("config/config.yaml")
        with open(config_path) as file:
            config = yaml.safe_load(file)

        self.s3_path = config["data_ingestion"]["s3_mri_path"]
        self.phenotype_url = config["data_ingestion"]["phenotypic_url"]
        self.mri_dir = Path(config["data_ingestion"]["mri_dir"])
        self.pheno_file = config["data_ingestion"]["phenotypic_file"]

    def create_dirs(self):

        self.mri_dir.mkdir(parents=True, exist_ok=True)

    def download_mri(self):

        logger.info("Downloading MRI dataset")

        command = [
            "aws",
            "s3",
            "sync",
            self.s3_path,
            str(self.mri_dir),
            "--exclude",
            "*",
            "--include",
            "*/mri/brain.mgz",
            "--no-sign-request",
        ]

        subprocess.run(command, check=True)

    def download_phenotype(self):

        logger.info("Downloading phenotypic dataset")

        command = [
            "curl",
            "-L",
            self.phenotype_url,
            "-o",
            self.pheno_file
        ]

        subprocess.run(command, check=True)

    def run(self):

        logger.info("Data Ingestion started")

        self.create_dirs()
        self.download_mri()
        self.download_phenotype()

        logger.info("Data Ingestion completed")