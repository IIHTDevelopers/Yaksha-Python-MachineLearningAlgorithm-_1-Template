import unittest
from test.TestUtils import TestUtils
import pandas as pd
import numpy as np
import os
from Linear_regression_auto import load_and_preprocess, prepare_data, train_and_save_model, evaluate_model
from titanic import *

class TestLinearRegressionAuto(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year']
        self.target = 'mpg'

    def test_load_and_preprocessauto(self):
        try:
            df = load_and_preprocess("auto-mpg.csv")
            expected_columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                                'acceleration', 'model-year']
            if isinstance(df, pd.DataFrame) and all(col in df.columns for col in expected_columns):
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", True, "functional")
                print("LinearRegressionAuto TestLoadAndPreprocess = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
                print("LinearRegressionAuto TestLoadAndPreprocess = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
            print(f"LinearRegressionAuto TestLoadAndPreprocess = Failed | Exception: {e}")

    def test_show_key_stats(self):
        try:
            df = load_and_preprocess("auto-mpg.csv")
            mean_displacement = df["displacement"].mean()
            min_horsepower = df["horsepower"].min()

            if round(mean_displacement, 2) == 193.65 and round(min_horsepower, 1) == 46.0:
                self.test_obj.yakshaAssert("TestShowKeyStats", True, "functional")
                print("LinearRegressionAuto TestShowKeyStats = Passed")
            else:
                self.test_obj.yakshaAssert("TestShowKeyStats", False, "functional")
                print("LinearRegressionAuto TestShowKeyStats = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestShowKeyStats", False, "functional")
            print(f"LinearRegressionAuto TestShowKeyStats = Failed | Exception: {e}")

    def test_prepare_data(self):
        try:
            df = load_and_preprocess("auto-mpg.csv")
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)

        # Basic type and structure checks
            valid_types = (isinstance(X_train, np.ndarray) and X_train.ndim == 2 and
                       isinstance(X_test, np.ndarray) and X_test.ndim == 2 and
                       isinstance(y_train, (pd.Series, np.ndarray)) and
                       isinstance(y_test, (pd.Series, np.ndarray)) and
                       hasattr(scaler, 'transform'))

        # Non-empty and consistent shapes
            valid_shapes = (X_train.shape[0] > 0 and X_test.shape[0] > 0 and
                        X_train.shape[0] == len(y_train) and
                        X_test.shape[0] == len(y_test))

            if valid_types and valid_shapes:
                self.test_obj.yakshaAssert("TestPrepareData", True, "functional")
                print("LinearRegressionAuto TestPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
                print("LinearRegressionAuto TestPrepareData = Failed: Invalid types or shapes")

        except Exception as e:
            self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
            print(f"LinearRegressionAuto TestPrepareData = Failed | Exception: {e}")

    def test_train_and_save_model(self):
        try:
            df = load_and_preprocess("auto-mpg.csv")
            X_train, _, y_train, _, _ = prepare_data(df, self.features, self.target)
            model_path = "linear_model.pkl"

            model = train_and_save_model(X_train, y_train, model_path)

            if os.path.exists(model_path) and hasattr(model, 'coef_'):
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", True, "functional")
                print("LinearRegressionAuto TestTrainAndSaveModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
                print("LinearRegressionAuto TestTrainAndSaveModel = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
            print(f"LinearRegressionAuto TestTrainAndSaveModel = Failed | Exception: {e}")

    def test_evaluate_model(self):
        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error

            df = load_and_preprocess("auto-mpg.csv")
            X_train, X_test, y_train, y_test, _ = prepare_data(df, self.features, self.target)

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Call the function you're testing
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Call the evaluation function (for its print output)
            evaluate_model(model, X_test, y_test)

            expected_preds = np.array([25.94750773, 30.75323514, 21.35934614, 26.86104935,
                                    29.34371694, 19.73673914, 7.76156783, 35.55021913,
                                    20.16105891, 28.90094774])

            if round(mse, 4) == 10.8164 and np.allclose(y_pred[:10], expected_preds, rtol=1e-4):
                self.test_obj.yakshaAssert("TestEvaluateModel", True, "functional")
                print("TestEvaluateModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
                print("TestEvaluateModel = Failed")

        except Exception as e:
                self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
                print(f"TestEvaluateModel = Failed | Exception: {e}")



class TestTitanic(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()

    def test_load_and_prepare_data(self):
        try:
            df = load_and_prepare_data("titanic.csv")
            expected_columns = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
            if (isinstance(df, pd.DataFrame) and all(col in df.columns for col in expected_columns) and
                    df['sex'].dtype == 'int64' and df['embarked'].dtype == 'int64'):
                self.test_obj.yakshaAssert("TestLoadAndPrepareData", True, "functional")
                print("Titanic TestLoadAndPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPrepareData", False, "functional")
                print("Titanic TestLoadAndPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPrepareData", False, "functional")
            print(f"Titanic TestLoadAndPrepareData = Failed | Exception: {e}")

    def test_explore_data(self):
        try:
            df = load_and_prepare_data("titanic.csv")
            fare_max = df['fare'].max()
            fare_std = df['fare'].std()

            if round(fare_max, 4) == 512.3292 and round(fare_std, 2) == 49.69:
                self.test_obj.yakshaAssert("TestExploreData", True, "functional")
                print("Titanic TestExploreData = Passed")
            else:
                self.test_obj.yakshaAssert("TestExploreData", False, "functional")
                print("Titanic TestExploreData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestExploreData", False, "functional")
            print(f"Titanic TestExploreData = Failed | Exception: {e}")

    def test_sigmoid_demo(self):
        try:
            result = sigmoid_demo()
            if round(result, 4) == 0.5:
                self.test_obj.yakshaAssert("TestSigmoidDemo", True, "functional")
                print("Titanic TestSigmoidDemo = Passed")
            else:
                self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
                print(f"Titanic TestSigmoidDemo = Failed | Got {round(result, 4)}")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
            print(f"Titanic TestSigmoidDemo = Failed | Exception: {e}")

    def test_femalescount(self):
        try:
            # Load the data first
            df = load_and_prepare_data("titanic.csv")
            
            result = count_females(df)

            # Check if the result matches the expected count
            expected_count = 314  # The expected count of females
            if result == expected_count:
                self.test_obj.yakshaAssert("Titanic femalescount", True, "functional")
                print("Titanic femalescount = Passed")
            else:
                self.test_obj.yakshaAssert("Titanic femalescount", False, "functional")
                print("Titanic femalescount = Failed")
        except Exception:
            self.test_obj.yakshaAssert("Titanic femalescount", False, "functional")
            print("Titanic femalescount = Failed")

    def test_train_and_evaluate(self):
        try:
            import numpy as np
            from sklearn.model_selection import train_test_split

            # Load and prepare the Titanic data
            df = load_and_prepare_data("titanic.csv")
            features = ['pclass', 'sex', 'age', 'fare', 'embarked']
            X = df[features]
            y = df['survived']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Call the function being tested
            y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, path="titanic_log_model.pkl")

            # Define expected output for first 10 predictions (may need updating based on model and data)
            expected_classes = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

            if isinstance(y_pred, np.ndarray) and np.array_equal(y_pred[:10], expected_classes):
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", True, "functional")
                print("Titanic TestTrainAndEvaluate = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
                print("Titanic TestTrainAndEvaluate = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
            print(f"Titanic TestTrainAndEvaluate = Failed | Exception: {e}")


if __name__ == '__main__':
    unittest.main()
