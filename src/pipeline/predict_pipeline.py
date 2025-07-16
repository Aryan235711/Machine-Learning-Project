import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Handle missing values and format inputs
            features = features.replace({None: np.nan})  # Explicitly replace None with np.nan

            # Imputer expects string dtype for object columns
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            for col in categorical_columns:
                if col in features.columns:
                    features[col] = features[col].astype(str).str.strip().str.lower()
                    features[col] = features[col].replace('nan', np.nan)  # Convert 'nan' strings back to proper NaN

            print(f"Features before transform:\n{features.head()}")
            print(f"Columns with missing values:\n{features.isnull().sum()}")


            #Scaling the data
            data_scaled = preprocessor.transform(features)
            #Making predictions
            prediction = model.predict(data_scaled)

            logging.info('Scaling and Prediction completed')
            return prediction

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        gender = str,
        race_ethnicity = str,
        parental_level_of_education = str,
        lunch = str,
        test_preparation_course = str,
        reading_score = int,
        writing_score = int
    ):
      self.gender = gender
      self.race_ethnicity = race_ethnicity
      self.parental_level_of_education = parental_level_of_education
      self.lunch = lunch
      self.test_preparation_course = test_preparation_course
      self.reading_score = reading_score
      self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        

        