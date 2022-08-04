class DataCleaner:
    def run(self, data):
        """
        perform cleaning process for input data

        Parameters
        ----------
        data : pd.DataFrame
            input data

        Returns
        -------
        data : pd.DataFrame
        """
        print("Step 1: remove duplicates")
        self.remove_duplicates(data)
        print("Step 2: check for na")
        self.na_check(data)
        
        return data
    
    def na_check(self, data):
        print(f"total number of missing values in the table: {data.isna().sum().sum()}")
        
    def remove_duplicates(self, data):
        rows = len(data)
        data.drop_duplicates(inplace=True)
        print(f"percentage of duplicated rows removed: {round(100*(rows - len(data)) / rows, 4)}%")
