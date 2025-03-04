# XAI-Lung-Disease-Detection
In this code, worked on Lung disease detection using XAI model Shap. 
import numpy as np
import pandas as pd


# Load dataset from drive
file_path = '/content/drive/MyDrive/port-dataset/lung_disease_data.csv'
df = pd.read_csv(file_path)

# Display basic info
df.head()
# Check dataset info
df.info()

# Check for missing values
df.isnull().sum()

# Summary statistics
df.describe()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Recovered'] = label_encoder.fit_transform(df['Recovered'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Disease Type'] = label_encoder.fit_transform(df['Disease Type'])

# Fill missing values in numerical columns with their mean
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Fill missing values in categorical columns with their mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Check if missing values are handled
print(df.isnull().sum())  # Should print 0s if all missing values are filled

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['Age', 'Lung Capacity', 'Hospital Visits']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

from sklearn.model_selection import train_test_split

X = df.drop('Disease Type', axis=1)
y = df['Disease Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of datasets
X_train.shape, X_test.shape, y_train.shape, y_test.shape
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_cols)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# Split data again after encoding
X = df.drop('Recovered', axis=1)  # Replace with actual target column
y = df['Recovered']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest again
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
!pip install shap  # Install SHAP if not already installed

import shap
import matplotlib.pyplot as plt
import numpy as np
print(X_train.dtypes)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
X_train = X_train.astype({'Smoking Status_1': 'int', 'Treatment Type_1': 'int', 'Treatment Type_2': 'int'})
X_test = X_test.astype({'Smoking Status_1': 'int', 'Treatment Type_1': 'int', 'Treatment Type_2': 'int'})
X_test = X_test[X_train.columns]
import shap

# Initialize SHAP TreeExplainer for RandomForest
explainer = shap.TreeExplainer(model)

# Compute SHAP values for test data
shap_values = explainer.shap_values(X_test)

# Summary plot to visualize feature importance
shap.summary_plot(shap_values, X_test)
