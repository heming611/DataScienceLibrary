import os, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt
from time import sleep
from iqlclient import IQLClient
from iqlclient.exceptions import (IQLNoShards, IQLServerError)
from dateutil import parser
from pyhive import hive
import logging
import pwd
import pandas as pd
import requests_kerberos
import pyhive
from pyhive import presto
from multiprocessing.dummy import Pool
from typing import List

data_dir = '../data'
resources_dir = '../resources'



class DataUtil:
    
    def merge(self, df_1, df_2, on, how):
        '''
        df_1, df_2: pandas dataframes
        on: List[str]
        how: 'inner', 'left', 'outer'
        '''
        prejoin_check(df_1, df_2, keys=on)
        df = pd.merge(df_1, df_2, on=on, how=how)
        num_of_rows_1, num_of_rows_2 = df_1.shape[0], df_2.shape[0]
        num_of_rows = df.shape[0]
        print('percentage of data retained: {:.2%}'.format(num_of_rows/num_of_rows_1))

        return df
    
    def prejoin_check(self, df_1, df_2, keys):
        '''
        functionality: print shapes and check if the join key is unique in both dataframes
        keys: List[str]
        '''
        print("lengths of first and second dfs: ({}, {})".format(len(df_1), len(df_2)))
        first, second = df_1.drop_duplicates(subset=keys), df_2.drop_duplicates(subset=keys)
        if len(first)!=len(df_1):
            print("Be careful, join key in first dataframe not unique!")
        if len(second)!=len(df_2):
            print("Be careful, join key in second dataframe not unique!")    
        if len(first)==len(df_1) and len(second)==len(df_2):
            print("join key unique!")


def from_64_to_32(df, columns=[], excluded_columns=[]):
    '''
    functionality: convert 64 bytes columns to 32 bytes columns to save space
    
    columns: List[str]: columns to convert
    excluded_columns: List[str]: columns not to convert
    '''
    
    if len(columns)>0:
        for col in columns:
            if str(df[col].dtypes) == "int64":
                df[col] = df[col].astype("int32")
            if str(df[col].dtypes) == "float64":
                df[col] = df[col].astype("float32")
    else: 
        for i, j in zip(df.columns, df.dtypes):
            if i not in excluded_columns:
                if str(j) == "int64":
                    df[i] = df[i].astype("int32")
                if str(j) == "float64":
                    df[i] = df[i].astype("float32")
            else:
                continue
               
    return df


def two_dataframe_equal_up_to_row_orders(df_1, df_2):
    df_1 = df_1.sort_values(by=df_1.columns.tolist()).reset_index(drop=True)
    df_2 = df_2.sort_values(by=df_2.columns.tolist()).reset_index(drop=True)
    
    return df_1.equals(df_2) 



def get_next_date(current_date, increment):
    '''
    functionality: get next date for increment days ahead of current_date
    '''
    return_date = (pd.Timestamp(current_date)+pd.Timedelta(days=increment)).date()
    
    return str(return_date)

def data_sanity_check(condition):
    '''
    condition: a boolean variable
    '''
    if condition:
        print("This data sanity check passed!")
    else:
        print("This data sanity check did not pass!")
    
    return

def check_row_keys_uniqueness(dataframe: pd.core.frame.DataFrame, keys: List[str]):
    
    df_tmp = dataframe[keys].drop_duplicates()

    if df_tmp.shape[0]==dataframe.shape[0]:
        print("{i} are unique keys for each row of the data.".format(i=keys))
    else:
        raise Exception("{i} are not unique keys for each row of the data, consider proper deduplication!".format(i=keys))
    del df_tmp
    
    
def is_this_unique_key(df, keys):
    '''
    check if a list of strings is the unique key of a dataframe
    df: a dataframe
    keys: List[str]
    '''
    tmp = df[keys].drop_duplicates()
    return tmp.shape[0]==df.shape[0]