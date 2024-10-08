import pandas as pd
import numpy as np

import os
import sys

import pickle
from sklearn.metrics import roc_auc_score
from src.exception import CustomException
from src.logger import logging


def ks(data=None, target=None, prob=None, asc=False):
    '''
    calculates Kolmogorov-Smirnov (KS) Statistic 
    '''
    try:
        logging.info("KS statistics started")
        data['target0'] = 1 - data[target]
        data['bucket'] = pd.qcut(data[prob], 10, duplicates='raise')
        grouped = data.groupby('bucket', as_index=False)
        kstable = pd.DataFrame()
        kstable['min_prob'] = grouped.min()[prob]
        kstable['max_prob'] = grouped.max()[prob]
        kstable['events'] = grouped.sum()[target]
        kstable['nonevents'] = grouped.sum()['target0']
        kstable['total'] = grouped.count()['target0']
        kstable['bucket_event_rate'] =kstable.events/kstable.total
        kstable = kstable.sort_values(by="min_prob", ascending=asc).reset_index(drop=True)
        kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
        kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
        kstable['cum_eventrate'] = (kstable.events / data[target].sum()).cumsum()
        kstable['cum_noneventrate'] = (kstable.nonevents / data['target0'].sum()).cumsum()
        kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100

        # Formating
        kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
        kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)

        # kstable.index = range(1, len(kstable)+1)
        kstable.index = range(1, 11)
        kstable.index.rename('Decile', inplace=True)
        logging.info("KS statistics completed")
        return kstable
    except Exception as e:
        raise CustomException(e, sys)


#ensure that objects are seen as categorical
def object_to_category(df):
    '''
    Ensure that obj is convertated to category to avoid error
    '''
    try:
        df = df.apply(lambda col: col.astype('category') if col.dtypes == 'object' else col)
    except Exception as e:
        raise CustomException(e,sys)


# save objects like trained models
def object_saver(file_path, obj):
    '''Save model 
    '''
    try:
        logging.info("model saving started")
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("model saving started")
    except Exception as e:
        raise CustomException(e, sys)
    
    
def object_loader(file_path):
    '''
    Load saved model 
    '''
    try:
        logging.info("model loading started")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        logging.info("model loading Completed")
    except Exception as e:
        raise CustomException(e, sys)

 # combine categoriacl and numerical features of df   
def filter_col(df:pd.DataFrame,cat:list,num:list):
    try:
        cat_col = [col for col in df.columns if col in cat]
        num_col = [col for col in df.columns if col in num]
        
        df1 = df[cat_col]
        df2 = df[num_col]
        
        d = pd.concat([df1,df2], axis=1)
        
        return d
    except Exception as e:
        raise CustomException(e, sys)
    

