import os, re, io, json, requests, pickle, logging
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np

# graphing
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import plotly.offline
from pandas_profiling import ProfileReport

# processing
from multiprocessing.dummy import Pool
from tqdm import tqdm
from tqdm.notebook import tqdm
tqdm.pandas()

# date and time
from time import time, sleep
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dateutil import parser

# permission
import requests_kerberos
#import plus
#import pg8000
import getpass


# math/stats
from scipy.stats import pearsonr
from math import ceil
import dc_stat_think as dcst

# data base
from pyhive import hive, presto
import pyhive

# machine learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import auc, roc_curve, mean_squared_error, precision_recall_curve, average_precision_score, plot_precision_recall_curve, plot_roc_curve, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.utils import shuffle, resample
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# model explanation
import shap

# others
import pwd
from typing import List
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')
