import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score, mean_squared_error


from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_squared_error
from functools import partial
rmse = partial(mean_squared_error, squared=False)

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test import data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            
            # Create a pipeline for the linear regression model with recursive feature elimination
            pipe_rfe = Pipeline(
                steps=[
                ('selector', RFECV(Ridge(alpha=100))),
                ('estimator', Ridge(alpha=100))
            ])
           
           # Define the parameter grid for the grid search
            param_grid_rfe = {
                'selector__step': [1, 5],
                'selector__cv': [5, 10]
            }
            
            # Perform a grid search to find the best hyperparameters and feature selector
            grid_rfe = GridSearchCV(pipe_rfe, param_grid=param_grid_rfe, n_jobs=-1)
            grid_rfe.fit(X_train, y_train)
            
            # Create a new pipeline with the best feature selector and fit on the training data
            best_pipe_rfe = Pipeline(
                steps=[
                ('selector', grid_rfe.best_estimator_.named_steps['selector']),
                ('estimator', Ridge(alpha=100,))
            ])
            
            best_pipe_rfe.fit(X_train, y_train)
            
            logging.info("Recursive selector trained")
            
            logging.info("Getting the best feature name from recursive selector")
            
            best_features = best_pipe_rfe['selector'].get_feature_names_out().tolist()
            
            print(best_features)
            
            X_train = X_train[best_features]
            
            X_test = X_test[best_features]
            
            logging.info("Best features ready")
            
            
            # Define the individual models
            random_forest_model = RandomForestRegressor(bootstrap=False, max_depth=None, max_features=0.5,
                                                        min_samples_split=10, min_samples_leaf=1, n_estimators=100)
            gradient_boosting_model = GradientBoostingRegressor(learning_rate=0.17, max_depth=7, n_estimators=150)

            
            # Create the dictionary with the models
            models = {
                "Random Forest": random_forest_model,
                "Gradient Boosting": gradient_boosting_model,
                "Voting Regressor": VotingRegressor([('Random Forest', random_forest_model), ('Gradient Boosting', gradient_boosting_model)])
            }
            
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            ##To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            ##To get best model name from the list
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset ")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            
            r2_score = r2_score(y_test,predicted)
            
            return r2_score
        
        
        except Exception as e:
            raise CustomException(e,sys)