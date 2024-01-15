import json
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
sns.set()
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt




global __model, __preprocessor
__model = None
__preprocessor = None

class LogScaling(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self   

    def transform(self, X):
        return np.log1p(X)

class TransformationPipeline:

    def __init__(self) -> None:
        pass
    
    def preprocess(self):
        cat_cols = ["brand", "processor_brand", "processor_name", "processor_gnrtn", "ram_gb", "ram_type", "ssd", "hdd", "os", "os_bit", "graphic_card_gb", "weight", "warranty", "touchscreen", "msoffice", "rating", "number of ratings", "number of reviews"]
        num_cols = ['Number of Ratings', 'Number of Reviews']

        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ]
        ) 

        cat_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]
        )

        preprocessor = ColumnTransformer([
            ("log_transform", LogScaling(), num_cols),
            ("num_pipeline", num_pipeline, num_cols),
            ("cat_pipelines",cat_pipeline,cat_cols)
            ], remainder= 'passthrough')
        
        self.__preprocessor = preprocessor
        return preprocessor 

def predict_price(**feature_values):
    
    load_saved_artifacts()
    # Create a DataFrame with default values
    data = {
        'brand': ['ASUS'],
        'processor_brand': ['Intel'],
        'processor_name': ['Core i3'],
        'processor_gnrtn': ['10th'],
        'ram_gb': ['4 GB'],
        'ram_type': ['DDR4'],
        'ssd': ['0 GB'],
        'hdd': ['1024 GB'],
        'os': ['Windows'],
        'os_bit': ['64-bit'],
        'graphic_card_gb': ['0 GB'],
        'weight': ['Casual'],
        'warranty': ['No warranty'],
        'Touchscreen': ['No'],
        'msoffice': ['No'],
        'rating': ['2 stars'],
        'Number of Ratings': [42],
        'Number of Reviews': [5]
    }

    # Update values with provided feature_values
    for key, value in feature_values.items():            
        if key in data:
            data[key][0] = value
            print(value)

    # Convert the dictionary to a DataFrame
    input_data = pd.DataFrame(data)

    # Preprocess the input data
    input_transformed = __preprocessor.transform(input_data)

    # Make predictions
    predicted_price = __model.predict(input_transformed)
    print(f'Predicted Price: ${predicted_price[0]*0.012:.2f}')
    return round(predicted_price[0],2)

def load_saved_artifacts():
    global __model, __preprocessor
    with open("server/artifacts/laptop_price_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    with open("server/artifacts/laptop_preprocessor.pickle", 'rb') as f:
        __preprocessor = pickle.load(f)
    print("Loading saved model...done")
   



if __name__ == '__main__':
    load_saved_artifacts()
    #predicted_price = predict_price(brand = 'HP', processor_brand = 'Intel', processor_name = 'Core i3', processor_gnrtn = '11th', ram_gb = '8 GB', ram_type = 'DDR4', ssd = '256 GB', hdd = '0 GB', os = 'Windows', os_bit = '64-bit', graphic_card_gb = '0 GB', weight	= 'Casual', warranty = '1 year', Touchscreen = 'Yes', msoffice = 'No', rating = '3 stars')
