from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
import os

# Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np




classification_pipeline = Pipeline(
    [
        ('MeanImputation', pp.MeanImputer(variables= config.NUM_FEATURES)),
        ('ModeImputatin', pp.ModeImputer(variables= config.CAT_FEATURES)),
        ('DomainProcessing', pp.DomainProcessing(variable_to_modify= config.FEATURE_TO_MODIFY, variable_to_add=config.FEATURE_TO_ADD)),
        ('DropFeaturs', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransformation', pp.LogTransform(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticRegression', LogisticRegression(random_state=0))
    ]
)