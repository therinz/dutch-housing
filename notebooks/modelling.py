# (c) 2020 Rinze Douma

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, median_absolute_error
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.regressor import ResidualsPlot, PredictionError


class DataFrameModel:
    BASE = os.path.join(os.pardir, "data")

    def __init__(self, filename):
        # Declare variables
        self.X_train = self.X_test = self.y_train = self.y_test = pd.DataFrame

        # Open file
        self.df = pd.read_pickle(os.path.join(DataFrameModel.BASE, filename))

        # Temp solution for incorrect column names
        self.df = (self.df.rename(columns={"rf_plat dak": "rf_plat_dak",
                                           "address_x": "address"})
                   .reset_index(drop=True))
        self.split_dataset()

        # Scale X
        self.scaler()

    def split_dataset(self, test_size=.4):
        """Return train & test set for X and y."""

        # Drop columns that appear highly correlated with other factors.
        # distractors = ["vve_kvk", "vve_am", "vve_per_contr",
        #               "vve_reserve_fund", "rt_pannen", "rf_plat_dak",
        #               "address", "price_m2"]

        # Set variables
        X = self.df[[col
                     for col in self.df.columns
                     if col not in ["asking_price", "address", "price_m2"]]]
        y = self.df["asking_price"]

        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(X, y,
                                                       test_size=test_size,
                                                       random_state=7)

    def scaler(self):
        """Convert numeric columns into scaled values."""

        # Select column names of factors with more than 2 values
        num_cols = [col for col in self.X_train.columns
                    if self.X_train[col].nunique() > 2
                    and self.X_train[col].dtype in ["int64", "float64"]]

        # Reset index for Null value issue
        for xf in [self.X_train, self.X_test]:
            xf.reset_index(drop=True, inplace=True)

        # Fit model
        std = StandardScaler()
        scaled_fit = std.fit(self.X_train[num_cols])

        # Apply
        for xf in [self.X_train, self.X_test]:
            scaled_X = scaled_fit.transform(xf[num_cols])
            xf[num_cols] = pd.DataFrame(scaled_X, columns=num_cols)

    def visualize_model(self, plot, model):
        """Visualize the data for a model in a certain plot."""

        visualizer = plot(model)
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        visualizer.show()

    def evaluate_model(self, model, viz=False):
        """Run ML model and return score"""
        models = {"LR": linear_model.LinearRegression,
                  "DT": DecisionTreeClassifier,
                  "RF": RandomForestClassifier}

        ml_model = models[model]()
        ml_model.fit(self.X_train, self.y_train)

        print(f"\n-----{str(model)}-----\n\n")

        if viz:
            # Create residuals plot
            self.visualize_model(ResidualsPlot, ml_model)

        predictions = ml_model.predict(self.X_test)
        acc = median_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        print(f"Model achieved an mean absolute error of {acc:.3f}."
              f"\nR2 score is {r2:.3f}")


"""
def model_func(filename):
    Open a file and run various models.

    # Open file
    df = pd.read_pickle(os.path.join(BASE, filename))

    # Temp solution for incorrect column names
    df = (df.rename(columns={"rf_plat dak": "rf_plat_dak",
                             "address_x": "address"})
          .reset_index(drop=True))
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Scale X
    X_train, X_test = scaler(X_train, X_test)

    models = [linear_model.LinearRegression,
              DecisionTreeClassifier,
              RandomForestClassifier]
    for model in models:
        evaluate_model(X_train, X_test, y_train, y_test, model)
"""

