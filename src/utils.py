import os
import pandas as pd
import numpy as np
import sys
import dill #dill is used for serializing Python objects, similar to pickle but more robust for complex objects
from src.logger import logging

from src.exception import CustomException

def save_object(obj, file_path):
    """
    Save an object to a file using pandas.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving object: {e}", sys)