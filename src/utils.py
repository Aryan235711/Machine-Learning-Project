# Utils.py is used to handle helper functions in the projects
# It contains functions for saving objects, evaluating models, evaluating metrics, etc.
import os
import pandas as pd
import numpy as np
import sys
import dill #dill is used for serializing Python objects, similar to pickle but more robust for complex objects
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from src.logger import logging
from src.exception import CustomException

def save_object(obj, file_path):
    """
    Save a Python object to a file using dill for serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving object: {e}", sys)

def calculate_metrics(y_test, y_pred, X_test):
    """
    Calculate the metrics for the model predictions.
    """
    try:
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'Adjusted R2_Score': r2_score(y_test, y_pred) - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        }
        return metrics
        
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, tuning_params=None):
    """
    Evaluate multiple models and return a report of their performance.
    """
    try:
        
        model_report = {}
        best_params = {}

        #Using GRidSearchCV for hyperparameter tuning
        for model_name, model in models.items():
            regressor = GridSearchCV(estimator=model,
                                     param_grid=tuning_params.get(model_name, {}),
                                     cv=3,
                                     n_jobs=-1)
            regressor.fit(X_train, y_train)
            
            logging.info(f"Best parameters for {model_name} : {regressor.best_params_}")

            best_params[model_name] = regressor.best_params_


        # Fit the models with the best parameters and evaluate them
        for model_name, model in models.items():
            if model_name in best_params.items():
                model.set_params(**best_params[model_name])
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            logging.info(f'{model_name} model trained and predictions made. Going for metrics calculations')
            report = calculate_metrics(y_test, y_pred, X_test)
            model_report[model_name] = {
                'model_object': model,  # Store the actual model object here as a value
                'metrics': report       # Store the metrics dictionary here
                }
            
            logging.info(f'{model_name} model evaluation completed and metrics report generated')
        
        # logging.info(f'Metrics report for all models: {model_report}')
            
        return model_report
    
    except Exception as e:
        raise CustomException(f"Error evaluating models: {e}", sys)


