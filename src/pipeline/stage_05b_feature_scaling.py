from src.components.feature_scaling import FeatureScaling
from src.utils.logger import logger


def run_stage():

    try:
        logger.info("Stage: Feature Scaling started")

        scaler = FeatureScaling()
        scaler.run()

        logger.info("Stage: Feature Scaling completed")

    except Exception as e:
        logger.error(e)
        raise


if __name__ == "__main__":
    run_stage()