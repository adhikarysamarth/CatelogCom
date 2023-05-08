## Group Members:
# Arpita Shrivas (02075452)
# Samarth Adhikary (01943125)
# Yash Alpeshbhai Patel (02071032)


# Import relevant libraries and functions
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from dmba import plotDecisionTree, classificationSummary, regressionSummary
from sklearn import tree


# Import the csv file using pandas and store it in a dataframe named 'df'
df = pd.read_csv("CatelogCom.csv", header=0)

# Examine the dataset variables
df.head()
df.columns
df.info() 
df.describe()
df.shape 

###############################################---> 1 <---###############################################

# Create training and validation sets
X = df.drop(['Purchase','Spending'],axis=1) # We drop the target variables and store the rest in X, also called input varibales or feature variables or predictors
y = df['Purchase'] # y stores the target variable or dependent variable or response variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# Define parameters for GridSearchCV

param_grid = {
    'max_depth': [3,5,7,9],
    'min_samples_split': [2,4,6],
    'min_samples_leaf': [1,2,5]
}

# Create decision tree model 
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

# Use GridSearch to find optimized tree
gridSearch = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1)
gridSearch.fit(X_train,y_train)


# Print the hyperparameters
print("Best Parameters: ", gridSearch.best_params_)
print("Best Score: ", gridSearch.best_score_)

# Predict y for training and testing data using the fitted linear model
y_pred_test_1 = gridSearch.predict(X_test)
y_pred_train_1 = gridSearch.predict(X_train)

cm_train = confusion_matrix(y_train, y_pred_train_1)
cm_test = confusion_matrix(y_test, y_pred_test_1)

print("Confusion Matrix (Train Data):\n", cm_train)
print("Confusion Matrix (Test Data):\n", cm_test)



###############################################---> 2 <---###############################################



# For this we need to subset the dataset where Purchase = 1

df2 = df[df['Purchase']==1]

# Create training and validation sets

X = df2.drop(['Purchase','Spending'],axis=1)
y = df2['Spending']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Check training and testing data
data={'Data Set':['X_train', 'X_test','y_train','y_test'], 'Shape': [X_train.shape, X_test.shape, y_train.shape, y_test.shape]}
df_split=pd.DataFrame(data)
df_split

# Assign linear regression function to lm 

lm = LinearRegression()

# Fit the model to the training dataset

lm.fit(X_train, y_train)

# Print Coefficients

print(pd.DataFrame({'Predictor': X.columns, 'Coefficient': lm.coef_}))
print('Intercept :', lm.intercept_)

# Predict y for training and testing data using the fitted linear model
y_pred_test_2 = lm.predict(X_test)
y_pred_train_2 = lm.predict(X_train)

# Print performance measures of the training data
regressionSummary(y_train, y_pred_train_2)

# Make predictions of Spending variable for the entire testing set
result = pd.DataFrame({'Predicted': y_pred_test_2, 'Actual': y_test, 'Residual': y_test - y_pred_test_2})
result.head(10)


# Print performance measures on the test data
regressionSummary(y_test, y_pred_test_2)

# Print Mean Squared Error and R-squared
print("Mean squared error: ", mean_squared_error(y_test, y_pred_test_2))
print("R2: ",r2_score(y_test, y_pred_test_2))




































