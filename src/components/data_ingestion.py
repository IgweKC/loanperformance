import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import object_to_category,filter_col
import sys
import os
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
warnings.filterwarnings('ignore')

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    payment_data_path: str=os.path.join("artifacts","payment.csv")
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def init_split_data_Ingestion(self):
        logging.info("Started the data ingestion")
        
        try:
            df = pd.read_csv("data/raw/train_loan_data.csv")
            logging.info("Read training data as dataframe")
           

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header = True)

            logging.info("Spliting train test")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info("Ingestion and split completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException (e, sys)




    def data_transformer_ingest(self):
        '''
        For reading train and test files 
        '''
        logging.info("Started the data ingestion")
        
        try:
            # will drop laon_ID after using it for computing a new feature
            numerica_feature =[
                'loan_id','principal', 'total_owing_at_issue','application_number',
                'applying_for_loan_number','loan_number', 'employee_count', 'paid_late', 
                'total_recovered_on_time','total_recovered_15_dpd', 'cash_yield_15_dpd'
                ]
            

            categorical_feature = [
                'acquisition_channel', 'sector',
                'applying_for_loan_number','payment_status',
                'approval_status', 'paid_late', 'Target'
                ]
            

            train_df = pd.read_csv("data/raw/train_loan_data.csv")
            test_df = pd.read_csv("data/raw/test_loan_data.csv")
            payemnt_df = pd.read_csv("data/raw/train_payment_data.csv")

            train_df = pd.read_csv("data/raw/train_loan_data.csv")
            
            train_df = filter_col(train_df,numerica_feature,categorical_feature)
            test_df = filter_col(test_df,numerica_feature,categorical_feature)
            
            logging.info("Read training and testing data as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.payment_data_path), exist_ok=True)
            
            
            train_df= object_to_category(train_df)
            test_df= object_to_category(test_df)

            traind = train_df.copy()
            testd = test_df.copy()

            train_df = self.feature_engineer(traind,payemnt_df)
            test_df =self.feature_engineer(testd,payemnt_df)


            test_df['Target'] = np.where(test_df['payment_status']=="Written off",1,0)
            test_df.drop('payment_status', axis=1, inplace=True)
            train_df = self.make_train_Target(train_df)
  

            train_df.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False, header=True)           
            payemnt_df.to_csv(self.ingestion_config.payment_data_path, index=False, header = True)

            logging.info("Ingestions with no split completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.payment_data_path

            )
        except Exception as e:
            raise CustomException (e, sys)
        
     
     
     # Feature Engineering
    def feature_engineer(self, df, train_payment_df):
        ''' 
        This function engineers a feature based on repayment duration (in day)
        '''
        try:

            logging.info("Feature Engineering started")
            
            # convert to date
            train_payment_df['paid_at']= pd.to_datetime(train_payment_df['paid_at'])
            
            # Calculate payment_count  for each loan_id in train_payment_df. Consider Deposit
            payment_count = train_payment_df[train_payment_df['transaction_type'] == 'Deposit']
            payment_count = payment_count.groupby('loan_id')['amount'].count().reset_index().rename(columns={'amount': 'payment_count'})
           
            # Calculate repayment_duration_days: difference between first and last payment dates
            first_payment_date = train_payment_df.groupby('loan_id')['paid_at'].min().reset_index().rename(columns={'paid_at': 'first_payment_date'})
            last_payment_date = train_payment_df.groupby('loan_id')['paid_at'].max().reset_index().rename(columns={'paid_at': 'last_payment_date'})
            df = df.merge(first_payment_date, on='loan_id', how='left').merge(last_payment_date, on='loan_id', how='left')
           
            # Calculate duration in days
            df['repayment_duration_days'] = (df['last_payment_date']-df['first_payment_date']).dt.days
            df['repayment_duration_days'].fillna(np.max(df['repayment_duration_days']), inplace=True)
            
            # Drop intermediate date columns to keep dataset clean
            df.drop(['first_payment_date', 'last_payment_date'], axis=1, inplace=True)
            
            logging.info("Feature Engineering Completed")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    # Use the computed average in combination with Business Knowledge to decide the Target
    def make_train_Target(self, df):
        '''
        Make Targets values: Default = 1, Not = 0
        '''
        try:
            logging.info("Making Target Started")
            df['Target'] = np.where(
                ((df['paid_late'] == True) | (df['cash_yield_15_dpd'] < 0) 
                | (df['total_recovered_on_time'] < df['total_recovered_15_dpd'])) , 1,0)
            logging.info("Making Target Started")
            return df
        except Exception as e:
            raise CustomException(e, sys)
            
    

if __name__=="__main__":
    obj=DataIngestion()
    obj.data_transformer_ingest()
