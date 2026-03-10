from src.components.feature_selection import FeatureSelection
from src.utils.logger import logger

def run_stage():
    try:
        logger.info("Stage 5: Feature Selection started")
        selector = FeatureSelection()
        selector.run()
        logger.info("Stage 5 completed")
    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        raise
if __name__ == "__main__":
    run_stage()