import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("C:/Users/riddh/OneDrive/Desktop/Project 1 ML/House Price Prediction Dataset - House Price Prediction Dataset.csv")

print(df)
#data acquistion
print(df.head)
print(df.dropna(inplace=True))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])
df=df.drop('Id',axis=1)
X=df.drop('Price',axis=1)
Y=df['Price']
X.shape
Y.shape
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in X.select_dtypes(include='object').columns:
    X[col] = le.fit_transform(X[col])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


df["Location"] = le.fit_transform(df["Location"])


df["Condition"] = le.fit_transform(df["Condition"])

df["Garage"] = le.fit_transform(df["Garage"])
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, Y_pred)
print("R2 Score:", r2)
df['Area'].corr(df['Price'])
df.corr(numeric_only=True)