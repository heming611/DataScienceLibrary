from package import *
from constants import *

class EDAUtilClass:
    
    def correlation_heatmap(self, data, save_to_path, figsize=(10, 6)):
        """
        plot heatmap of correlation of a given dataframe

        Parameters
        ----------
        data : pd.DataFrame
            input data

        Returns
        -------
        """
        plt.figure(figsize = figsize)
        corr_matrix = round(data.corr(), 2)
        mask = np.triu(np.ones_like(corr_matrix, dtype = np.bool), k = 1)
        heatmap = sns.heatmap(corr_matrix, mask = mask, vmin = -1, vmax = 1, annot = True)
        heatmap.set_title('Correlation Heatmap', fontdict = {'fontsize':15}, pad = 10)
        plt.tight_layout()
        plt.savefig(fname=save_to_path, bbox_inches="tight")
        plt.show()
        
    def row_percentage_by_categorical_feature(self, data, feature_col):
        """
        For categorical feature/target, count number/percentage of rows by value of the feature/target
        """
        output = data.groupby([feature_col]).size().reset_index()
        output.rename(columns={0:"row_count"}, inplace=True)
        output["row_percentage"] = round(100*output["row_count"]/len(data))
        
        return output

    def variation_of_target_by_feature(
        self, 
        data, 
        feature_col, 
        target_col, 
        num_row, 
        num_col, 
        figsize=(12,10), 
        xlim="auto", 
        ylim="auto",
        feature_type="categorical",
        target_type="categorical",
        sup_title=None,
        save_to_path=None
    ):
        """convert an input string time from a time zone to a string time to another time zone

        Parameters
        ----------
        feature_col : str
            feature column name
        target_col : str
            target column name
        feature_type : str
            either be 'categorical' or 'continuous'
        target_type : str
            either be 'categorical' or 'continuous'

        Returns
        -------
        str
            a string time in the to_zone
        """
        type_set = set(["categorical", "continuous"])
        if feature_type not in type_set or target_type not in type_set:
            raise ValueError("feature and target type can only take values 'continuous' or 'categorical'!")
            
        if feature_type == "categorical":
            # distribution or histogram of target conditional on feature
            feature_values = data[feature_col].unique()
            plt.figure(figsize=figsize)
            for i, value in enumerate(feature_values):
                df_tmp = data[data[feature_col]==value]
                plt.subplot(num_row, num_col, i+1)
                sns.histplot(df_tmp[target_col]).set_title("feature = "+str(value)+f" (count: {df_tmp.shape[0]})")
                if xlim!="auto":
                    plt.xlim(xlim[0], xlim[1])
                if ylim!="auto": 
                    plt.ylim(ylim[0], ylim[1])  

        else: # feature is continuous
            if target_type == "categorical":
                # mean or distribution of feature conditional on target
                target_values = data[target_col].unique()
                plt.figure(figsize=figsize)
                for i, value in enumerate(target_values):
                    df_tmp = data[data[target_col]==value]
                    plt.subplot(num_row, num_col, i+1)
                    sns.histplot(df_tmp[feature_col]).set_title("target = "+str(value)+f" (count: {df_tmp.shape[0]})")
                    if xlim!="auto":
                        plt.xlim(xlim[0], xlim[1])
                    if ylim!="auto": 
                        plt.ylim(ylim[0], ylim[1])
                     
            else:
                x, y = data[feature_col], data[target_col]
                corr = pearsonr(x, y)
                plt.figure(figsize=figsize)
                sns.scatterplot(x=x, y=y)
                if xlim!="auto":
                    plt.xlim(xlim[0], xlim[1])
                if ylim!="auto": 
                    plt.ylim(ylim[0], ylim[1])  
        plt.suptitle(sup_title)  
        plt.tight_layout()
        plt.savefig(fname=save_to_path, bbox_inches="tight")
        
        
