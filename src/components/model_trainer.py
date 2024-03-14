import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }
            
            model_report = self.eval_model(X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test, models= models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]               
            
            if best_model_score < 0.6 :
                raise CustomException("Model score is less than 0.6. No best model", sys)
            logging.info(f"Best Model: {best_model_name} with score: {best_model_score}")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= models[best_model_name],
            )
            
            predicted= models[best_model_name].predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            
            return r2_score_value
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def eval_model(self,X_train, y_train, X_test, y_test, models):
        try:
            model_report = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                
                y_train_pred = model.predict(X_train)
                
                y_test_pred = model.predict(X_test)
                
                train_model_score= r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                model_report[model_name] = test_model_score
            return model_report
        
        except Exception as e:
            raise CustomException(e, sys)