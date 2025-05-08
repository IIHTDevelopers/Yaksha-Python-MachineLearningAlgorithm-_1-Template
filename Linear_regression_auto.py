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

    # Return empty DataFrame to avoid errors but make tests fail
    return pd.DataFrame()


# Function 2: Perform EDA (Only Max and Std for Fare)
def explore_data(df):
    # TODO: Calculate the maximum value of the 'fare' column
    # TODO: Calculate the standard deviation of the 'fare' column
    # TODO: Print " Fare - Max: {max_fare}, Std Dev: {std_fare:.2f}"
    max_fare = None   # Replace with df['fare'].max()
    std_fare = None  # Replace with df['fare'].std()
    return round(max_fare, 4), round(std_fare, 2)


# Function 3: Sigmoid activation for a single value
def sigmoid_demo():
    # TODO: Set z = 0
    # TODO: Calculate sigmoid = 1 / (1 + np.exp(-z))
    # TODO: Print " Sigmoid(0) = {sigmoid:.4f}"
    return sigmoid_demo()

def count_females(df):
        # TODO: write the function to  count number of females in the dataset
        # TODO:Assuming 'female' was encoded as 0
    count=None #write you logic here
    return count

# Function 5: Train and evaluate model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="titanic_log_model.pkl"):
    # TODO: Create a LogisticRegression model with max_iter=1000
    # TODO: Fit the model with the training data
    # TODO: Save the model using joblib.dump()
    # TODO: Print " Model trained and saved to '{path}'"
    # TODO: Use the model to predict classes for X_test
    # TODO: Use the model to predict probabilities for X_test and get the positive class probability
        # TODO: Print "🔍Sample Predictions:" followed by the first 10 predictions
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
