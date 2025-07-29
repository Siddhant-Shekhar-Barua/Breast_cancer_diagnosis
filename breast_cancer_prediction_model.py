# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 20:05:06 2025

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from  sklearn.ensemble import RandomForestClassifier   # Machine learning model
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score

df = pd.read_csv(r'C:\Users\HP\Documents\python\data.csv')
print('Dimensions:', df.shape)
print('data_type:',df.dtypes)
print('missing_values:')
print(df.isnull())
print('-------The data is clean-----------')


df.drop("id", axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

sns.countplot(x='diagnosis', data=df)
plt.title("Malignant vs Benign Tumors")
plt.xlabel("Diagnosis (0=Benign, 1=Malignant)")
plt.ylabel("Count")
plt.show()

X = df.drop('diagnosis', axis=1)  # All the input features
y = df['diagnosis'] 

X_train ,X_test , y_train , y_test  = train_test_split(X,y , test_size= 0.2 , random_state= 42)

model_rf = RandomForestClassifier()
model_rf.fit(X_train ,y_train)
y_pred = model_rf.predict(X_test)

print('confusion_matrix:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#feature importance

feat_importance  = pd.Series(model_rf.feature_importances_, index=X.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title('top 10 features that impact the diagnosis')
plt.show()





