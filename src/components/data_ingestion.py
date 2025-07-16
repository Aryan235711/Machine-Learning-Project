# data_ingestion.py is used to ingest the data from the source and save it in the artifacts folder
# It includes functions for reading the data, splitting it into train and test sets, and saving the preprocessed data
import os
from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
# from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')

        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as a pandas dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=23)

            logging.info('Train test split completed')
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    # Combining the data ingestion and transformation process
    logging.info('Starting data ingestion and transformation process')
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))
        
