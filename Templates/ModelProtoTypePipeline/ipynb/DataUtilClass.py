from package import *
from constants import *

class DataUtilClass:
    
    def merge(self, df_1, df_2, on, how):
        """
        df_1, df_2: pandas dataframes
        on: List[str]
        how: 'inner', 'left', 'outer'
        """
        self.prejoin_check(df_1, df_2, keys=on)
        df = pd.merge(df_1, df_2, on=on, how=how)
        num_of_rows_1, num_of_rows_2 = df_1.shape[0], df_2.shape[0]
        num_of_rows = df.shape[0]
        print('percentage of data retained: {:.2%}'.format(num_of_rows/num_of_rows_1))

        return df
    
    def prejoin_check(self, df_1, df_2, keys):
        """
        functionality: print shapes and check if the join key is unique in both dataframes
        keys: List[str]
        """
        print("lengths of first and second dfs: ({}, {})".format(len(df_1), len(df_2)))
        first, second = df_1.drop_duplicates(subset=keys), df_2.drop_duplicates(subset=keys)
        if len(first)!=len(df_1):
            print("Be careful, join key in first dataframe not unique!")
        if len(second)!=len(df_2):
            print("Be careful, join key in second dataframe not unique!")    
        if len(first)==len(df_1) and len(second)==len(df_2):
            print("join key unique!")