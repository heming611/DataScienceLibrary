from package import *
from constants import *
      
class PredictorUtilClass:
    
    def get_predictions(self, X, y, model):
        data = pd.concat([X, y], axis=1)
        X = xgb.DMatrix(X)
        data.rename(columns={"target": "y"}, inplace=True)
        data["y_hat"] = model.predict(X)
           
        return data
