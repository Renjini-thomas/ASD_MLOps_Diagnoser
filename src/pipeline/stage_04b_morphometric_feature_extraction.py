from src.components.morphometric_feature_extraction import MorphometricFeatureExtraction
from src.utils.logger import logger


class MorphometricFeatureExtractionPipeline:

    def __init__(self):
        self.morph_stage = MorphometricFeatureExtraction()

    def run(self):

        try:
            logger.info("Stage 04b: Morphometric Feature Extraction started")

            self.morph_stage.run()

            logger.info("Stage 04b: Morphometric Feature Extraction completed")

        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == "__main__":

    pipeline = MorphometricFeatureExtractionPipeline()
    pipeline.run()