import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


# Function 1: Load and preprocess the dataset
def load_and_preprocess(path):
    # TODO: Load the CSV file using pandas
    # TODO: Convert column names to lowercase and strip whitespace
    # TODO: Drop rows with missing values
    # TODO: Print " Data loaded and cleaned."

    return pd.DataFrame()


# Function 2: Show mean of displacement and min of horsepower
def show_key_stats(df):
    # TODO: Calculate the mean of the 'displacement' column
    # TODO: Find the minimum value of the 'horsepower' column
    # TODO: Print the mean displacement :eg " Mean Displacement: {value:.2f}"
    # TODO: Print the minimum horsepower :eg" Minimum Horsepower: {value}"
    pass


# Function 3: Prepare data for training
def prepare_data(df, features, target):
    # TODO: Extract features (X) and target (y) from the DataFrame
    # TODO: Create a StandardScaler and fit_transform the features
    # TODO: Split the data into training and testing sets (test_size=0.2, random_state=42)
    # TODO: Print " Data prepared and split."

    # Expected format is 2D array, dont use 1D array instead
    return np.array([]), np.array([]), pd.Series(), pd.Series(), StandardScaler()


# Function 4: Train the model and save it
def train_and_save_model(X_train, y_train, model_path="linear_model.pkl"):
    # TODO: Create a LinearRegression model
    # TODO: Fit the model with the training data
    # TODO: Save the model using joblib.dump()
    # TODO: Print " Model trained and saved to '{model_path}'"

    # Return an untrained model to avoid errors but make tests fail
    return LinearRegression()


# Function 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    # TODO: Use the model to predict values for X_test
    # TODO: Calculate the mean squared error between predictions and actual values
    # TODO: Print " Mean Squared Error: {mse:.4f}"
    # TODO: Print " Sample Predictions:" followed by the first 10 predictions
    pass


# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year']
    target = 'mpg'

    # TODO: Call load_and_preprocess() with "auto-mpg.csv"
    # TODO: Call show_key_stats() with the loaded DataFrame
    # TODO: Call prepare_data() with the DataFrame, features, and target
    # TODO: Call train_and_save_model() with the training data
    # TODO: Call evaluate_model() with the model and testing data
    pass
