from package import *
from constants import *

class DataCollectorUtilClass:
    
    def __init__(self, connection):
        self.connection = connection
        
    def run(self, start_date_train_data):
        start_time = time()
        print(f"Step 1: download data 1 starting from {start_date_train_data}")
        self.get_data_1(start_date_train_data)
        print(f"execution time in mins: {(time()-start_time)//60}")
        
        start_time = time()
        print("Step 2: download data 2")
        self.get_data_2()
        print(f"execution time in mins: {(time()-start_time)//60}")
        
        print("Step 3: close snowflake connector")
        self.connection.close()
              
    def get_data_1(self, start_date_train_data):
        query = f"""
                SELECT A, B, C
                FROM table_1
                WHERE date > '{start_date_train_data}'  
                """

        df = pd.read_sql(query, self.connection)
        df.drop_duplicates(inplace=True)
        print(f"data_1 shape: {df.shape}")
        df.to_parquet(f"../data/data_1.parquet", index=False)
        
    def get_data_2(self):
        query = """
                SELECT A, B, C
                FROM table_2
                """
        df = pd.read_sql(query, self.connection)
        print(f"data_2 shape: {df.shape}")
        df.to_parquet(f"../data/data_2.parquet", index=False)
        