# use other helper classes 
from src.exception import CustomException
from src.logger import logging
from src.utils import ks
from src.utils import object_to_category

#systems and os
import sys
import os

# basics
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from src.utils import object_saver
from dataclasses import dataclass

# no printing warning
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    model_path: str=os.path.join("artifacts","model.pkl")
    result_path: str=os.path.join("evaluation","KS_metrics.csv")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def init_param_tuning(self, train_df,test_df):
        try:

            logging.info("Model parameter tuning started")

            params = {
                'min_child_weight': [1, 5, 10],
                'max_depth': [3, 4],
                'learning_rate' : [0.001, 0.01,0.1],
                'colsample_bytree': [0.1, 0.2],
            }


            clf = lgb.LGBMClassifier(
                verbose_eval=False,
                objective = 'binary',
                boosting_type = 'gbdt',
                seed= 0,
                verbose= -1,
                metric = 'auc',
                nthread = 16,
            )
            
            folds = 3

            ## stratified sampling
            skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
            selected_features = train_df.drop(["Target"], axis= 1, inplace=False).columns
            selected_features

            # convert to cat
            train_df = object_to_category(train_df)
            test_df = object_to_category(test_df)

            # Target
            target = "Target"
            selected_train_features = train_df.drop([target], axis= 1, inplace=False).columns
            selected_test_features = test_df.drop([target], axis= 1, inplace=False).columns

            # X_train , X_test, y_train, y_test = train_test_split(data_features, data_target, test_size=0.01, random_state=42)
            X_train = train_df[selected_train_features]
            X_test = test_df[selected_test_features]
            y_train = train_df[target].tolist()
            y_test = test_df[target].tolist()

            grid_search = GridSearchCV(clf, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3 )

            # Search the best param
            grid_search.fit(X_train, y_train)

            logging.info("Model parameter tuning done")
            return (X_train,
                    X_test,
                    y_train,
                    y_test,
                    grid_search
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def init_model_trainer(self, X_train,X_test,y_train, y_test,grid_search):
        try:
            logging.info("Model Training and Evaluation started")

            os.makedirs(os.path.dirname(self.model_trainer_config.result_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.model_trainer_config.model_path), exist_ok=True)

            classifier = grid_search.best_estimator_
            
            # fit and evaluate model
            classifier.fit(X_train,y_train)
            y_train_probs = classifier.predict_proba(X_train)[:, 1]
            y_test_probs = classifier.predict_proba(X_test)[:, 1]
            
            #test perfromance on train
            train_auc = roc_auc_score(y_train, y_train_probs)
            test_auc = roc_auc_score(y_test, y_test_probs)
            #print(f"Model results. Train AUC : {train_auc}. Test AUC : {test_auc}")

            #kolmogorov-smirnov statistics
            test_result_df = pd.DataFrame({"target" : y_test , "proba": y_test_probs })
            summary_table = ks(test_result_df,"target","proba")

            ks_result = pd.DataFrame(summary_table)
            ks_result.to_csv(self.model_trainer_config.result_path,index=False, header=True)
           
            # save model
            object_saver(self.model_trainer_config.model_path, classifier)
        

            logging.info("Training and Evaluation done")

            return self.model_trainer_config.result_path
        
        except Exception as e:
            raise CustomException (e, sys)
        
    def initiate_model_trainer(self):
        ''' 
        Run the model and send the result to evaluation folder
        '''
        try: 
            train_df = pd.read_csv("artifacts/train.csv")
            test_df = pd.read_csv("artifacts/test.csv")

            X_train,X_test,y_train,y_test,Grid_search =  self.init_param_tuning(train_df,test_df)
            self.init_model_trainer(X_train,X_test,y_train,y_test,Grid_search)

        except Exception as e:
            raise CustomException (e, sys)
        

if __name__=="__main__":
    obj=ModelTrainer()
    obj.initiate_model_trainer()