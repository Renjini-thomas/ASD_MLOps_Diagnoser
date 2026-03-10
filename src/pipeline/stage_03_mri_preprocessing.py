from src.components.mri_preprocessing import MRIPreprocessing
from src.utils.logger import logger
def run_stage():
    try:

        print("Stage 3: MRI Preprocessing started")
        logger.info("Stage 3: MRI Preprocessing started")
        processor = MRIPreprocessing()
        processor.run()

        print("Stage 3 completed")
        logger.info("Stage 3 completed")
    except Exception as e:
        print(f"Stage 3 failed: {e}")
        logger.error(f"Stage 3 failed: {e}")
        raise
if __name__ == "__main__":
    run_stage() 