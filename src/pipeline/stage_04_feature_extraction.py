from src.components.feature_extraction import FeatureExtraction
from src.utils.logger import logger

def run_stage():
    try:
        print("Stage 4: Feature Extraction started")
        logger.info("Stage 4: Feature Extraction started")
        extractor = FeatureExtraction()
        extractor.run()
        print("Stage 4 completed")
        logger.info("Stage 4 completed")
    except Exception as e:  
        print(f"Stage 4 failed: {e}")
        raise
if __name__ == "__main__":
    run_stage() 