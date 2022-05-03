from package import *
from constants import *

class DataCleanerUtilClass:
    
    def run(self, data):
        #data = self.remove_customers_paying_multiple_currencies(data)
        
        print(f"data size: {data.shape}")
        print("Step 1: filter out rows with abnormal feature values")
        data = self.filter_by_features(data)
        print(f"data size: {data.shape}")
        
        print("Step 2: filter out rows with other criterion")
        data = self.other_filters(data)
        print(f"data size: {data.shape}")

        return data
    
    def filter_by_features(self, data):
        
        '''
        write custom code
        '''
        
        return data
        
    def other_filters(self, data):
        '''
        write custom code
        '''
        
        return output

