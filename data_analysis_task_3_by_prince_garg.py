
from google.colab import drive

drive.mount('/content.gdrive')  #connect drive to colab

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('/content.gdrive/My Drive/DATA - 3.csv')
df

#remove duplicates values
df = df.drop_duplicates()
df

df.describe()

#checking null values
df.isnull().sum()

#removing null values
df = df.dropna(subset = {"nativeLanguage"})
df = df.dropna(subset = {"city"})
df = df.dropna(subset = {"country"})
df = df.dropna(subset = {"R1"})
df = df.dropna(subset = {"R2"})
df = df.dropna(subset = {"R3"})

df

#visualization

sns.relplot(x="participantID", y="nativeLanguage", data =df)

#model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
model = LinearRegression()

y= df['participantID']
X = df[['age','participantID','education','responseID']]

df

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2529)

model.fit(X_train,y_train)

model.intercept_

model.coef_

#prediction
y_pred = model.predict(X_test)

#accuracy

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
mean_absolute_percentage_error(y_test,y_pred)

mean_absolute_error(y_test,y_pred)

mean_squared_error(y_test,y_pred)