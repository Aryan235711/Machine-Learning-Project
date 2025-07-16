# modeltrainer.py is used to train the ml models
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

#we define the config of the model trainer here but not inside the ModelTrainer class because this config is used to create the ModelTrainer class instance
#This way, we can easily change the config without modifying the class definition
#and we can also use this config in other parts of the code if needed
#This is a good practice to separate the configuration from the implementation
#And the dataclass decorator is used to automatically generate the __init__ method and other methods for the class
@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and testing input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info('Splitting completed')

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False),
                'XGBRegressor': XGBRegressor()
            }

            tuning_params = {
                'RandomForestRegressor': {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'DecisionTreeRegressor': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'LinearRegression': {},
                'KNeighborsRegressor': {
                    'n_neighbors': [3, 5, 7]
                },
                'CatBoostRegressor': {},
                'XGBRegressor': {}
            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, tuning_params=tuning_params)

            logging.info('Model evaluation completed')

            #to get the best model from the model report
            
            best_models = sorted({model_name : values['metrics']['r2_score'] for model_name, values in model_report.items()}, key=lambda x: x[1], reverse=True)
            logging.info('This is a list of best models sorted by r2_score: {}'.format(best_models))

            best_model = model_report[best_models[0]]['model_object']
            
            logging.info("Saving the best model: {}".format(best_model))
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info('Model saved successfully')

            predicted = best_model.predict(X_test)

            r2_score_value = r2_score(y_test, predicted)
            return r2_score_value
        

        except Exception as e:
            raise CustomException(e,sys)

