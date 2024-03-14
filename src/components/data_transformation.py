import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation: 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        '''
        This function is used to get the data transformation object.
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            
            categeorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )
            
            cat_pipeline = Pipeline(
                 steps=[
                     ("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot_encoder", OneHotEncoder()),
                     ("scalar", StandardScaler(with_mean=False))
                 ]
            )            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categeorical_columns)
                ]
            )
            logging.info("full pipeline of encoding and scaling done successfully")
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function is used to initiate the data transformation process.
        '''
        try:
            preprocessor_obj = self.get_data_transformer_obj()
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            target_column_name = 'math_score'
            
            X_train= train_data.drop(columns=[target_column_name], axis=1)
            y_train= train_data[target_column_name]
            
            X_test= test_data.drop(columns=[target_column_name], axis=1)
            y_test= test_data[target_column_name]
            
            train_data_transformed = preprocessor_obj.fit_transform(X_train)
            test_data_transformed = preprocessor_obj.transform(X_test)
            
            train_arr= np.c_[
                train_data_transformed,
                y_train
            ]
            test_arr= np.c_[
                test_data_transformed,
                y_test
            ]
            save_object(
                file_path= self.data_transformation_config.preprocessor_ob_file_path,
                obj= preprocessor_obj
            )
            logging.info("Data transformation done successfully")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
                
        except Exception as e:
            raise CustomException(e, sys)