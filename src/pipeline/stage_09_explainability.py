from src.components.explainability import ModelExplainability
from src.utils.logger import logger


if __name__ == "__main__":

    try:
        print("Stage 9: Explainability started")
        logger.info("Stage 9: Explainability started")

        explainability = ModelExplainability()
        explainability.run()

        print("Stage 9 completed")
        logger.info("Stage 9 completed")

    except Exception as e:
        print(f"Stage 9 failed: {e}")
        logger.error(f"Stage 9 failed: {e}")
        raise