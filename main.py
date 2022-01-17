import sklearn as sl
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

""" Data preprocessing """
# Import training data
trainingData = pd.read_csv("titanic/train.csv")

# independent variables
x = trainingData[['Pclass', 'Sex', 'Age']]

# Dependent variable
y = trainingData[['Survived']]

# Dealing with NaN inputs in numeric fields using the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[['Pclass','Age']])
x[['Pclass','Age']] = imputer.transform(x[['Pclass','Age']])

# Encoding dependent variable (catagorical data)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1])] , remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)