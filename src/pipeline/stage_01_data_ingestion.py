from src.components.data_ingestion import DataIngestion
from src.utils.logger import logger


def run_stage():

    try:

        logger.info("Stage 1: Data Ingestion started")

        ingestion = DataIngestion()
        ingestion.run()

        logger.info("Stage 1 completed")

    except Exception as e:

        logger.error(f"Stage 1 failed: {e}")
        raise
if __name__ == "__main__":
    run_stage()