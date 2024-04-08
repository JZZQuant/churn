from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import joblib


def pre_processing(df,target,p_f):
    print("Shape before transformation : " + str(df.shape[1]))
    # Find numerical and categorical columns
    numeric_features = df.select_dtypes(include=[np.float64]).columns.tolist()
    categorical_features = df.select_dtypes(include=[object]).columns.tolist()
    #Dump high cardinality categoric features and convert ordinal variables accordingly
    h_cardinal = [(x,df[x].nunique()) for x in df.columns if x not in numeric_features]
    h_card_features = []
    for _col,_cardinality in h_cardinal:
        if _cardinality >1000 :
            print("High Cardinality : "+_col)
            h_card_features.append(_col)
            # df.drop(columns= x,inplace =True)
        if _cardinality <=2:
            print("low cardinality : "+ _col)
            # df[x] = pd.factorize(df[_col])[0]

    preprocessor = ColumnTransformer(
    transformers=[
        ('drop_columns','drop',h_card_features),  # Drop all high cardinality columns column
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0))]), numeric_features)
    ])

    _out= pd.DataFrame(preprocessor.fit_transform(df))
    print("Shape after transformation : " + str(_out.shape[1]))
    joblib.dump(preprocessor, filename=p_f)

    return _out,target[0].to_numpy() ,categorical_features
