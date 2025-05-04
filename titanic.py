import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


# Function 1: Load, clean, and encode Titanic dataset
def load_and_prepare_data(path="titanic.csv"):
    # TODO: Load the CSV file using pandas
    # TODO: Fill missing values in 'sex' column with 'unknown'
    # TODO: Fill missing values in 'embarked' column with 'S'
    # TODO: Fill missing values in 'age' column with the median age
    # TODO: Fill missing values in 'fare' column with the median fare
    # TODO: Use LabelEncoder to encode 'sex' column
    # TODO: Use LabelEncoder to encode 'embarked' column
    # TODO: Print " Data loaded, cleaned, and encoded."
    
    
    return pd.DataFrame()


# Function 2: Perform EDA (Only Max and Std for Fare)
def explore_data(df):
    # TODO: Calculate the maximum value of the 'fare' column
    # TODO: Calculate the standard deviation of the 'fare' column
    # TODO: Print " Fare - Max: {max_fare}, Std Dev: {std_fare:.2f}"
    return round(max_fare, 4), round(std_fare, 2)

# Function 3: Sigmoid activation for a single value
def sigmoid_demo():
    # TODO: Set z = 0
    # TODO: Calculate sigmoid = 1 / (1 + np.exp(-z))
    # TODO: Print " Sigmoid(0) = {sigmoid:.4f}"
    return sigmoid_demo()


# Function 4: Custom cost function (Log Loss)
def cost_function(y_true, y_pred_prob):
    # TODO: Set epsilon to 1e-15 to avoid log(0)
    # TODO: Clip prediction probabilities between epsilon and 1-epsilon
    # TODO: Calculate binary cross-entropy using the applied log Loss formula 
    
    
    return 0.0


# Function 5: Train and evaluate model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="titanic_log_model.pkl"):
    # TODO: Create a LogisticRegression model with max_iter=1000
    # TODO: Fit the model with the training data
    # TODO: Save the model using joblib.dump()
    # TODO: Print " Model trained and saved to '{path}'"
    # TODO: Use the model to predict classes for X_test
    # TODO: Use the model to predict probabilities for X_test and get the positive class probability
    # TODO: Calculate the cost using the custom cost_function
    # TODO: Print " Log Loss (Custom Cost): {cost:.4f}"
    # TODO: Print " Sample Predictions:" followed by the first 10 predictions
    pass


# --------- Main Logic ---------
if __name__ == "__main__":
    # TODO: Call load_and_prepare_data() with "titanic.csv"
    # TODO: Call explore_data() with the loaded DataFrame
    # TODO: Call sigmoid_demo()
    # TODO: Define features list: ['pclass', 'sex', 'age', 'fare', 'embarked']
    # TODO: Extract features (X) and target (y) from the DataFrame
    # TODO: Split the data into training and testing sets (test_size=0.2, random_state=42)
    # TODO: Call train_and_evaluate() with the training and testing data
    pass
