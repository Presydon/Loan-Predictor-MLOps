import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os

# Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset
import joblib


classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred==1, 'Y', "N")
    result = {"predictions": output}
    return result

# testing
# def generate_predictions():
#     test_data = load_dataset(config.TEST_FILE)
#     pred = classification_pipeline.predict(test_data[config.FEATURES])
#     output = np.where(pred==1, 'Y', "N")
#     print(output)
#     #result = {"Predictions": output}
#     return output


if __name__ == "__main__":
    generate_predictions()