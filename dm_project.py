
# Import relevant libraries and functions
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

# Import the csv file using pandas and store it in a dataframe named 'df'
df = pd.read_csv("CatelogCom.csv", header=0)

# Examine the dataset variables
df.head()
df.columns
df.info() 
df.describe()

# Create training and validation sets
X = df.drop(['Purchase','Spending'],axis=1) # We drop the target variables and store the rest in X, also called input varibales or feature variables or predictors
y = df['Purchase'] # y stores the target variable or dependent variable or response variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Define parameters for GridSearchCV

param_grid = {
    'max_depth': [3,5,7,9],
    'min_samples_split': [2,4,6,8],
    'min_samples_leaf': [1,2,3,4,5]
}

# Create decision tree model 

dt = DecisionTreeClassifier(random_state=42)
gridSearch = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1)
gridSearch.fit(X_train,y_train)

print("Best Parameters: ", gridSearch.best_params_)
print("Best Score: ", gridSearch.best_score_)

# Test on performance on the test data
y_pred = gridSearch.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

