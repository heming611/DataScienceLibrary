from constants import *
from package import *

class TrainTestSplitUtilClass:
    def __init__(
        self, 
        test_size, 
        split_strategy, 
        time_series_sort_col=None,
        data_imbalance_strategy=None
    ):
        """
        split_strategy: 'time_series_split', 'random'
        """
        self.test_size = test_size
        self.split_strategy = split_strategy
        self.time_series_sort_col = time_series_sort_col
        self.data_imbalance_strategy = data_imbalance_strategy
    
    def run(self, data):
        print("Step 1: rearrange data either by sorting by time or shuffle !")
        data = self.rearrange_data(data)
        print("Step 2: train test split!")
        X_train, y_train, X_test, y_test = self.train_test_split(data)
        
        if self.data_imbalance_strategy:
            df_train = pd.concat([X_train, y_train], axis=1)
            df_train_majority = df_train[df_train[TARGET_COL]==0]
            df_train_minority = df_train[df_train[TARGET_COL]==1]
            
            if self.data_imbalance_strategy == "majority_down_sample":
                # Downsample majority class
                df_train_majority_downsampled = resample(
                    df_train_majority,
                    replace=False,    # sample without replacement
                    n_samples=len(df_train_minority),  # to match minority class
                    random_state=0
                )
                # Combine minority class with downsampled majority class
                df_train = pd.concat([df_train_majority_downsampled, df_train_minority])
                X_train, y_train = df_train[FEATURE_COLS], df_train[TARGET_COL]
            elif data_imbalance_strategy == "minority_up_sample":
                df_train_minority_upsampled = resample(
                    df_train_minority, 
                    replace=True,     # sample with replacement
                    n_samples=len(df_train_majority),    # to match majority class
                    random_state=0
                )
 
                # Combine majority class with upsampled minority class
                df_train = pd.concat([df_train_majority, df_train_minority_upsampled])
                X_train, y_train = df_train[u.FEATURE_COLS], df_train[u.TARGET_COL]
            elif self.data_imbalance_strategy == "smote":
                # Use SMOTE to oversample minority class and Then undersample the majority class
                over = SMOTE(sampling_strategy=0.5)
                under = RandomUnderSampler(sampling_strategy=1)
                steps = [('o', over), ('u', under)]
                pipeline = Pipeline(steps=steps)
                X_train, y_train = pipeline.fit_resample(X_train, y_train)

            else:
                raise ValueError("data_imbalance_strategy value not supported!")
        
        return X_train, y_train, X_test, y_test
    
    def rearrange_data(self, data):
        if self.split_strategy == "time_series_split":
            data.sort_values(by=[self.time_series_sort_col], inplace=True)
        elif self.split_strategy == "random":
            data = shuffle(data, random_state=0)
        else:
            raise ValueError("only 'time_series_split' and 'random' are valid values for split_strategy")
            
        return data
    
    def train_test_split(self, data):
        
        df_train, df_test = np.split(data, [int((1-self.test_size)*len(data))])
        X_train, y_train = df_train[FEATURE_COLS], df_train[TARGET_COL]
        X_test, y_test = df_test[FEATURE_COLS], df_test[TARGET_COL]
  
        return X_train, y_train, X_test, y_test
    
