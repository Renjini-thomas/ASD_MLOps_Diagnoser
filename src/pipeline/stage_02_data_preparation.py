from src.components.data_preparation import DataPreparation
from src.utils.logger import logger

def run_stage():
    try:
        print("Stage 2: Dataset Preparation started")
        logger.info("Stage 2: Dataset Preparation started")
        preparation = DataPreparation()
        preparation.run()
        logger.info("Stage 2 completed")
        print("Stage 2 completed")
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        raise
if __name__ == "__main__":
    run_stage()