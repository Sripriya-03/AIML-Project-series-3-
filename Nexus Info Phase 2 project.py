#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore harmless warnings 

import warnings 
warnings.filterwarnings("ignore")

# Set to display all the columns in dataset

pd.set_option("display.max_columns", None)

# Import psql to run queries 

import pandasql as psql


# In[2]:


import pandas as pd

# Load the medical dataset from an Excel file
file_path = r"C:\Users\ratho\OneDrive\Desktop\Medical_dataset(1).xlsx"
medical = pd.read_excel(file_path, header=0)

# Copy the file to a backup file
medical_BK = medical.copy()


# In[3]:


medical.head(10)


# In[4]:


medical.info()


# In[5]:


medical.nunique()


# In[6]:


medical['Family Medical History'].value_counts()


# In[7]:


#use labelencoder to handle categorical data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
medical['Family Medical History']=LE.fit_transform(medical[['Family Medical History']])


# In[8]:


medical['Family Medical History'].value_counts()


# In[11]:


#count the target or dependent variable by '0' &'1' and their proportion 
#(>=10:1,then the dataset is imbalance data) 
disease_count=medical['Family Medical History'].value_counts()
print("Class 0:",disease_count[0])
print("Class 1",disease_count[1])
print("Class 2",disease_count[2])
print("Class 3",disease_count[3])
print('Proportion:',round(disease_count[0]/disease_count[1]/disease_count[2]/disease_count[3],4),':1')
print('Total :',len(medical))


# In[12]:


# Displaying Duplicate values with in dataset
medical_dup= medical[medical.duplicated(keep='last')]

# Display the duplicate records

medical_dup


# In[13]:


medical.isnull().sum()


# In[14]:


del medical["Gender"]
del medical["Age"]
medical.head()


# In[15]:


medical.describe()


# In[16]:


# Identify the independent and Target (dependent) variables

IndepVar = []
for col in medical.columns:
    if col != 'Family Medical History':
        IndepVar.append(col)

TargetVar = 'Family Medical History'

x =medical[IndepVar]
y =medical[TargetVar]


# In[17]:


# Split the data into train and test (random sampling)

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Display the shape for train & test data

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[18]:


# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))
#since all x are indpendent variables
x_train = mmscaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test= mmscaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


# In[21]:


#load the result dataset 
medical_results=pd.read_csv(r"C:\Users\ratho\Downloads\knnresults.csv",header=0)
medical_results.head()


# In[28]:


pip install pandas numpy scikit-learn


# In[33]:


pip install imbalanced-learn xgboost openpyxl


# In[39]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load the dataset
file_path = r"C:\Users\ratho\OneDrive\Desktop\Medical_dataset(1).xlsx"
data = pd.read_excel(file_path, header=0)

# Preprocessing
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Family Medical History'] = le.fit_transform(data['Family Medical History'])

# Splitting features and target variable
X = data.drop('Family Medical History', axis=1)
y = data['Family Medical History']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning for Random Forest, XGBoost, and Gradient Boosting
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0]
}

# Models initialization with refined hyperparameter tuning
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),  # Adjust KNN parameters
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1, verbose=2),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'XGBoost': GridSearchCV(xgb.XGBClassifier(random_state=42), param_grid_xgb, cv=3, n_jobs=-1, verbose=2),
    'Gradient Boosting': GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3, n_jobs=-1, verbose=2)
}

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Evaluating each model
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    accuracy, report = evaluate_model(model, X_train_res, y_train_res, X_test, y_test)
    results[model_name] = {'Accuracy': accuracy, 'Classification Report': report}

# Displaying results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print("Classification Report:")
    print(result['Classification Report'])
    print("-" * 50)


# In[40]:


# Plotting bar graph for accuracies
model_names = list(results.keys())
accuracies = [results[model]['Accuracy'] for model in model_names]

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Models')
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




