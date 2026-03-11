from src.components.feature_aggregation import FeatureAggregation
from src.utils.logger import logger


def run_stage():
    try:
        logger.info("Stage 5: Feature Aggregation started")

        aggregator = FeatureAggregation()
        aggregator.run()

        logger.info("Stage 5: Feature Aggregation completed")

    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        raise


if __name__ == "__main__":
    run_stage()