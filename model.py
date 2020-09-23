# Graduate Admission Prediction Model using Linear Regression

# Importing the Libraries
import pandas as pd
from sklearn.model_selection import train_test_split # For splitting data into Training and Validation Set
from sklearn.linear_model import LinearRegression # For Linear Regression
import pickle # For saving data into a file

# Loading the dataset
dataset = pd.read_csv("Admission_Predict.csv")

#Preprocessing the data
dataset = dataset.drop(labels="Serial No.", axis = 1)
df = dataset.drop("GRE Score", axis =1)
df = df.drop("TOEFL Score", axis =1)

# Splitting data into Training and Testing
X = df.iloc[:, :-1].values
y = df.iloc[:, 5].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state =0)

# Applying Linear Regression Model
reg_linear = LinearRegression()
reg_linear.fit(X_train, y_train)
y_pred_linear = reg_linear.predict(X_valid)
y_pred_linear = (y_pred_linear*100)

# Saving the data columns from training
pickle.dump(reg_linear, open('model.pkl','wb'))
