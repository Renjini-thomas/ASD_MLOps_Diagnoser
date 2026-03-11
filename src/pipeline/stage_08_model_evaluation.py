from src.components.model_evaluation import ModelEvaluation
from src.utils.logger import logger
def run_stage():
    try:
        print("Stage 7: Model Evaluation started")
        logger.info("Stage 7: Model Evaluation started")
        evaluator = ModelEvaluation()
        evaluator.run()
        print("Stage 7 completed")
        logger.info("Stage 7 completed")
    except Exception as e:
        print(f"Stage 7 failed: {e}")
        logger.error(f"Stage 7 failed: {e}")
        raise
if __name__ == "__main__":
    run_stage() 