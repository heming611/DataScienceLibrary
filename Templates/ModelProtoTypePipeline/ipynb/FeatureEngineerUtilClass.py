from constants import *
from package import *

class FeatureEngineerUtilClass:
    
    def run(self, data):
        print(f"data size: {data.shape}")
        print("Step 1: create historical features")
        data = self.create_historical_features(data)
        
        print(f"data size: {data.shape}")
        print("Step 2: create dummy features")
        data = self.get_dummy(data)
        
        print(f"data size: {data.shape}")
        print("Step 3: impute missing values")
        data = self.impute_missing_values(data)
        
        data = self.create_label(data)
        return data
        
    
    def create_historical_features(self, data):
        '''
        write custom code
        '''

        return data
        
    def historical_features_udf(self, grp):
        '''
        write custom code
        '''
        return output
    
    def get_dummy(self, data):
        '''
        write custom code
        '''
        return data
    
    def impute_missing_values(self, data):
        '''
        write custom code
        '''
        print("missing values before na fill")
        print(data.isna().sum())
        data.fillna(value={"A": 1,
                           "B": 2,
                           "C": 3,
                           }, inplace=True)
        print("missing values after na fill")
        print(data.isna().sum())
        
        return data
    
    def create_label(self, data):
        '''
        write custom code
        '''
        return data
    