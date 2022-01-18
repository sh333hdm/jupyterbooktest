#Funktionen
import pandas as pd
import numpy as np
import geohash as gh

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from statsmodels.tools.eval_measures import mse, rmse


#Einlesen der Daten 
def read_data():
    path = "data\project_data.csv"
    df = pd.read_csv(path)
    return df

#Grundlegedende Transformationen der Daten 
def transform_data(df):
    # category variables
    df['ocean_proximity'] = df['ocean_proximity'].astype("category")
    df['price_category'] = df['price_category'].astype("category")
    #numeric variables
    df['median_house_value'] = pd.to_numeric(df['median_house_value'], errors='coerce')
    df['housing_median_age'] = pd.to_numeric(df['housing_median_age'], errors='coerce')
    #korrektur below/above
    df['price_category'] = np.where(df['median_house_value'] > 150000, 'above', 'below')
    df['price_category'] = df['price_category'].astype("category")
    return df

#Splitten der Daten in Trainings- und Testdaten 
def split_data(df):
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)
    return train_dataset, test_dataset

def fill_missingdata(df):
    median_total_bedrooms = df["total_bedrooms"].median()
    df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
    df.dropna(subset=['median_house_value', 'housing_median_age'], inplace = True)
    return df

def add_dummies(df):
    dummies = pd.get_dummies(df[['ocean_proximity', 'geohash']])
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=['ocean_proximity', 'geohash'], inplace= True)
    return df

def add_feautures(df): 
    df['households_population'] = df['households']/df['population']
    df['total_rooms_households'] = df['total_rooms']/df['households']
    df['total_rooms_total_bedrooms'] = df['total_rooms']/df['total_bedrooms']
    #geohash
    df_geo = df[['latitude', 'longitude']]
    df['geohash']=df_geo.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=3), axis=1)
    df['geohash'] = df['geohash'].astype("category")
    return df

def drop_outliners(df):
    id_income = df[df['median_income'] >= 15.0].index
    df.drop(index=id_income, inplace =True)
    id_housing = df[df['housing_median_age'] >= 52.0].index
    df.drop(index=id_housing, inplace =True)
    id_value = df[df['median_house_value'] >= 500000].index
    df.drop(index=id_value, inplace =True)
    return df

def print_metrics(df, predicted):
    # Header
    print('-'*50)
    print(f'Metrics for: {predicted}\n')
    
    # Confusion Matrix
    y_actu = pd.Series(df['price_category'], name='Actual')
    y_pred = pd.Series(df[predicted], name='Predicted')
    df_conf = pd.crosstab(y_actu, y_pred)
    print(df_conf)

    
    # Confusion Matrix to variables:
    pop = df_conf.values.sum()
    tp = df_conf['below']['below']
    tn = df_conf['above']['above']
    fp = df_conf['below']['above']
    fn = df_conf['above']['below']
    
    # Metrics
    accuracy = (tp + tn) / pop
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1_score:.4f} \n')

def build_preprocessor():
    #Transformieren von numeriscchen Werten
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()) 
        ])
    #Transformieren von kategorischen Variablen
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder( dtype= int, handle_unknown= "ignore")) 
        ])
    preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])
    return preprocessor

def metrics_sk(y_train, y_pred):
    r2 = round(r2_score(y_train, y_pred), 4)
    rmse = round(mean_squared_error(y_train, y_pred, squared = False), 2)
    mse = round (mean_squared_error(y_train, y_pred), 2)

    print("r2: ",r2)
    print("RMSE ",rmse)
    print("MSE: ", mse)
    
    return r2, rmse, mse

def metrics_stats(model, df):
    r2= round(model.rsquared,4)
    i_rmse = round(rmse(df['median_house_value'], df['y_pred']), 2)
    i_mse = round (mse(df['median_house_value'], df['y_pred']), 2)

    print("r2: ",r2)
    print("RMSE ",i_rmse)
    print("MSE: ",i_mse)
    
    return r2, i_rmse, i_mse