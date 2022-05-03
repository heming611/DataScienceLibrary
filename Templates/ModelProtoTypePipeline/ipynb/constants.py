# Directories
data_dir = '../data'
resources_dir = '../resources'

# Parameters Configurations

from datetime import date
from dateutil.relativedelta import relativedelta
START_DATE_TRAIN_DATA = (date.today() + relativedelta(months=-8)).strftime("%Y-%m-%d")

FEATURE_COLS = [
    "A","B","C"
]

TARGET_COL = "target"

BINARY_PREDICTION_NAME = "y_hat"
IDENTIFIER_COLS = ["id"]
# model
CV_FOLDS = 5
MODEL_PATH = f"../model/xgboost_classifier.json" 

# snowflask access
SNOWFLAKE_USER = "xyz"
SNOWFLAKE_ROLE = "xyz"
SNOWFLAKE_PASSPHRASE = "xyz"
SNOWFLAKE_WAREHOUSE = "xyz" 
SNOWFLAKE_PRIVATE_KEY = """xyz"""


