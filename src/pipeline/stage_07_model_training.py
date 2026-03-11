from src.components.model_training import ModelTraining
from src.utils.logger import logger

def run_stage():
    try:
        print("Stage 6: Model Training started")
        logger.info("Stage 6: Model Training started")
        trainer = ModelTraining()
        trainer.run()
        print("Stage 6 completed")
        logger.info("Stage 6 completed")
    except Exception as e:
        print(f"Stage 6 failed: {e}")
        logger.error(f"Stage 6 failed: {e}")
        raise
if __name__ == "__main__":
    run_stage()