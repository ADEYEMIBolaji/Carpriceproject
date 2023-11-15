import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

#import pipeline and transformers
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer

#import preprocessors 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer

from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score, mean_squared_error

import time


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_squared_error
from functools import partial
rmse = partial(mean_squared_error, squared=False)

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifact',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            categorical_features = ['standard_colour', 'standard_make', 'standard_model',
                                     'vehicle_condition','body_type',
                                    'crossover_car_and_van',
                                    'fuel_type','vehicle_ratings']
            numerical_features = ['mileage',
                                    'year_of_registration',
                                    'Advert_year',
                                    'Car_Age',
                                    'Avg_mileage']
            # Define your target columns
            # target_columns = ['standard_colour', 'standard_make', 'standard_model']
            # Create a pipeline
            # enc_target_pipeline = Pipeline(
            #     steps=[
            #     ('target_encoder', TargetEncoder(cols=target_columns, smoothing=0.2)),
            # ])
            
            #setting up a pipeline for scaling the numerical features
            numeric_pipeline = Pipeline(
                steps=[ 
                    ("scaler", StandardScaler())
                ]
            )
            
            #category encoding
            category_pipeline = Pipeline(
                steps=[
                ("onehot", OneHotEncoder(sparse=False, handle_unknown='infrequent_if_exist', min_frequency=0.01))
            ])
            
            logging.info("Categorical columns encoding completed")
            
            logging.info("Categorical columns Standard scaling completed")
            
            preprocessor = ColumnTransformer(
                [
                    # ("enc_target_pipeline",enc_target_pipeline,target_columns),
                    ("category_pipeline",category_pipeline,categorical_features),
                    ("numeric_pipeline",numeric_pipeline,numerical_features)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read Train and Test data completed")
            
            logging.info("obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name=["price","public_reference"]
            
            input_feature_train_df =train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df['price']
            
            input_feature_test_df =test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df['price']
            
            logging.info(
                f"Applying preprocessing object on trainin dataframe and testing dataframe"
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)