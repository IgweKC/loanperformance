from src.components.data_ingestion import DataIngestion
from src.pipeline.model_trainer_pipeline import ModelTrainer
from src.logger import logging
from src.exception import CustomException

import sys

if __name__=="__main_":
    try:
        logging.info("ML for loan performance started")
        
        # create data ingestion and model trainer instances
        data_ingestion_obj = DataIngestion()
        model_trainer_obj = ModelTrainer()

        # initiate them in sequence
        data_ingestion_obj.data_transformer_ingest()
        model_trainer_obj.initiate_model_trainer()

        logging.info("ML for loan performance Completed")
    except Exception as e:
       raise CustomException(e,sys)
