import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import object_loader


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            print("Before Loading")
            model=object_loader(file_path=model_path)
            
            # we would use processor pipeline if required
            print("After Loading")

            # At the moment, our tree based model is ok
            preds=model.predict(features)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

    def batch_predict(self,df):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            print("Before Loading")
            model=object_loader(file_path=model_path)
            # we would use processor pipeline here if required
            # At the moment, our tree based model is ok
            preds= model.predict_proba(df)[:, 1]
            return preds
        except Exception as e:
            raise CustomException(e,sys)    
        