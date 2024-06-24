import sys
import logging
from yaml import SafeLoader
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import create_log_path , CustomLogger

log_file_path = create_log_path('make_dataset')
# create the custom logger object
dataset_logger = CustomLogger(logger_name= 'make_dataset'
                              ,log_filename= log_file_path)

# set the level of logging to INFO
dataset_logger.set_log_level(level = logging.INFO)

def load_raw_data(input_path: Path)-> pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows , columns = raw_data.shape
    dataset_logger.save_logs(msg = f'{input_path.stem} data read having {rows} rows and {columns} olumns',
                             log_level = 'info')
    
    return raw_data

def train_val_split(data:pd.DataFrame,
                    test_size:float,
                    random_state: int)->tuple[pd.DataFrame , pd.DataFrame]:
    
    train_data , val_data = 