from src.pipeline.stage_01_data_ingestion import run_stage as ingestion_stage
from src.pipeline.stage_02_data_preparation import run_stage as preparation_stage
from src.pipeline.stage_03_mri_preprocessing import run_stage as preprocessing_stage
from src.pipeline.stage_04_feature_extraction import run_stage as feature_extraction_stage
from src.pipeline.stage_05_feature_selection import run_stage as feature_selection_stage
from src.pipeline.stage_06_model_training import run_stage as model_training_stage
from src.pipeline.stage_07_model_evaluation import run_stage as model_evaluation_stage
def main():

    # print("Starting ASD MLOps Pipeline")

    # ingestion_stage()

    # print("Pipeline completed")

    print("\nStage 2: Dataset Preparation")
    preparation_stage()

    
    # print("\nStage 3: MRI Preprocessing")
    # preprocessing_stage()

    # print("\nStage 4: Feature Extraction")
    # feature_extraction_stage()

    # print("\nStage 5: Feature Selection")
    # feature_selection_stage()
    # print("\nPipeline Completed")
    

    # print("\nStage 6: Model Training")
    # model_training_stage()  
    # print("\nPipeline Completed")

    # print("\nStage 7: Model Evaluation")
    # model_evaluation_stage()
    # print("\nPipeline Completed")
if __name__ == "__main__":
    main()