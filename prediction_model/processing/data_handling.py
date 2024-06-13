import pandas as pd
import os
import pandas
import joblib
from prediction_model.config import config


# Return dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data


# Serializing model
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f'Model has been serialized as {config.MODEL_NAME}')



# Deserializing model
def load_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print('Model has been deserialized')
    return model_loaded


