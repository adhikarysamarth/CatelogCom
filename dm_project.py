# Import relevant libraries and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, classification_report
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
    'max_depth': [3,5,7,9, None],
    'min_samples_split': [2,4,6],
    'min_samples_leaf': [1,2,5],
    'criterion' : ['gini','entropy']
}

# Create decision tree model 
dt = DecisionTreeClassifier(random_state = 42)

# Use GridSearch to find optimized tree
gridSearch = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gridSearch.fit(X_train,y_train)


# Print the hyperparameters
print("Best Parameters: ", gridSearch.best_params_)
print("Best Score: ", gridSearch.best_score_)

# Predict y for training and testing data using the fitted linear model
y_pred_test = gridSearch.predict(X_test)
y_pred_train = gridSearch.predict(X_train)

# Confusion Matrices
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

print("Confusion Matrix (Train Data):\n", cm_train)
print("Confusion Matrix (Test Data):\n", cm_test)

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Confusion Matrix - Train")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Confusion Matrix - Test")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred_test))


###############################################---> 2 <---###############################################



# For this we need to subset the dataset where Purchase = 1
df2 = df[df['Purchase']==1]

# Create training and validation sets
X1 = df2.drop(['Purchase','Spending'],axis=1)
y1 = df2['Spending']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.2,random_state=42)

# Assign linear regression function to lm 
lm = LinearRegression()

# Fit the model to the training dataset
lm.fit(X1_train, y1_train)

# Print Coefficients
print(pd.DataFrame({'Predictor': X1.columns, 'Coefficient': lm.coef_}))
print('Intercept :', lm.intercept_)

# Predict y for training and testing data using the fitted linear model
y1_pred_test_2 = lm.predict(X1_test)
y1_pred_train_2 = lm.predict(X1_train)

# Print performance measures of the training data
regressionSummary(y1_train, y1_pred_train_2)

# Create DataFrame with predicted, actual, and residual values
result = pd.DataFrame({
    'Predicted': y1_pred_test_2,
    'Actual': y1_test.values,
    'Residual': y1_test.values - y1_pred_test_2
})

# Display the first few rows
print(result.head())


# Print performance measures on the test data
regressionSummary(y1_test, y1_pred_test_2)

# Print Mean Squared Error and R-squared
print("Mean squared error: ", mean_squared_error(y1_test, y1_pred_test_2))
print("R2: ",r2_score(y1_test, y1_pred_test_2))

# Residual plot
residuals = y1_test - y1_pred_test_2
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30, color='purple')
plt.title("Residuals Distribution (Test Set)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()



































