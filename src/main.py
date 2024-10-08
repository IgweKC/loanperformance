#Basic
import pandas as pd
from sklearn.metrics import recall_score, precision_score

#local
from src.components.data_ingestion import DataIngestion
from src.pipeline.model_trainer_pipeline import ModelTrainer
from src.pipeline.model_predict_pipeline import PredictPipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import object_loader

#sys and os
import sys
import os

def test_main():
    try:
        logging.info("ML for loan performance started")
        
        print("Beging Main")
        # create data ingestion and model trainer instances
        data_ingestion_obj = DataIngestion()
        model_trainer_obj = ModelTrainer()
        model_predict = PredictPipeline()

        # initiate them in sequence
        data_ingestion_obj.data_transformer_ingest()
        model_trainer_obj.initiate_model_trainer()

        
        print("Beging Eval")
        logging.info("ML for loan performance Completed")
    except Exception as e:
       raise CustomException(e,sys)


if __name__=="__main__":
    test_main()