# (c) 2020 Rinze Douma

import os
import importlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, median_absolute_error
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.regressor import ResidualsPlot, PredictionError

importlib.import_module("helpers")
from helpers import validate_input                  # noqa
importlib.import_module("json_dataframe")
from json_dataframe import APARTMENTS               # noqa


class MachineLearnModel:
    BASE = os.path.join(os.pardir, "data")

    def __init__(self, filename, mode="apartments"):
        # Declare variables
        self.X_train = self.X_test = self.y_train = self.y_test = pd.DataFrame

        # Open file
        self.df = pd.read_pickle(os.path.join(MachineLearnModel.BASE,
                                              filename))

        # Check for null values
        if any(self.df.isnull().any()):
            raise ValueError("Null values present in DataFrame.")

        # Temp solution for incorrect column names
        self.df = (self.df.rename(columns={"rf_plat dak": "rf_plat_dak"})
                   .drop(columns=["xf_attic"])
                   .reset_index(drop=True))
        # Temp solution to drop where asking price is 0
        self.df.drop(self.df[self.df["asking_price"] == 0].index,
                     inplace=True)

        # Drop columns that appear highly correlated with other factors.
        # distractors = ["vve_kvk", "vve_am", "vve_per_contr",
        #               "vve_reserve_fund", "rt_pannen", "rf_plat_dak",
        #               "address", "price_m2"]
        distractors = ["address", "price_m2", "service_fees_pm"]
        self.df.drop(columns=distractors, inplace=True)

        # Drop outliers for asking price
        self.df.drop(self.df[(self.df.asking_price > 10000000)
                             | (self.df.asking_price < 100000)]
                     .index, inplace=True)

        # Select apartments or houses
        if mode.lower() == "apartments":
            self.apartments()
        else:
            self.houses()

        # Split into X and y
        self.split_dataset()

        # Scale X
        self.scaler()

    def apartments(self):
        """Remove outliers and drop non-apartment columns."""

        ap = list(self.df[(self.df["asking_price"] > 10000000)
                          | (self.df["asking_price"] < 100000)].index)
        yr = list(self.df[self.df["build_year"] < 1750].index)
        sf = list(self.df[self.df["service_fees_pm"] > 500].index)
        m3 = list(self.df[(self.df["property_m3"] < 10)
                          | (self.df["property_m3"] > 800)].index)
        br = list(self.df[self.df["num_bathrooms"] > 4].index)
        tl = list(self.df[self.df["num_toilets"] > 3].index)
        bed = list(self.df[self.df["bedrooms"] > 5].index)
        do = list(self.df[self.df["days_online"] > 120].index)

        # Drop rows that have outliers
        self.df = self.df.drop(ap + yr + sf + m3 + br + tl + bed + do)

        # Drop columns that are not relevant for apartments
        pt = [col
              for col in self.df.columns
              if (col.startswith("pt") and col not in APARTMENTS)
              or (col.startswith("rt_") or col.startswith("rf_"))]
        self.df = self.df.drop(columns=pt + ["land_m2", "floors"])

    """def houses(self):
        Remove outliers and drop apartment columns.



        # Drop columns that are not relevant for houses
        pt = [col
              for col in self.df.columns
              if col in APARTMENTS
              or (col.startswith("rt_") or col.startswith("rf_")))]
        self.df = self.df.drop(columns=pt + []))"""

    def split_dataset(self, test_size=.4):
        """Return train & test set for X and y."""

        # Set variables
        X = self.df[[col
                     for col in self.df.columns
                     if col != "asking_price"]]
        y = self.df["asking_price"]

        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(X, y,
                                                       test_size=test_size,
                                                       random_state=7)

    def scaler(self):
        """Convert numeric columns into scaled values."""

        # Select column names of factors with more than 2 values
        num_cols = [col for col in self.df.columns
                    if self.df[col].nunique() > 2
                    and self.df[col].dtype in ["int64", "float64"]
                    and col != "asking_price"]

        # Fit mdl
        std = StandardScaler()
        scaled_fit = std.fit(self.X_train[num_cols])

        # Apply to dataframe, train set first
        self.X_train.reset_index(drop=True, inplace=True)
        scaled = pd.DataFrame(scaled_fit.transform(self.X_train[num_cols]),
                              columns=num_cols)
        self.X_train = self.X_train.drop(columns=num_cols, axis=1)
        self.X_train = self.X_train.merge(scaled,
                                          left_index=True,
                                          right_index=True,
                                          how="outer")
        # test set
        self.X_test.reset_index(drop=True, inplace=True)
        scaled_test = pd.DataFrame(scaled_fit.transform(self.X_test[num_cols]),
                                   columns=num_cols)
        self.X_test = self.X_test.drop(columns=num_cols, axis=1)
        self.X_test = self.X_test.merge(scaled_test,
                                        left_index=True,
                                        right_index=True,
                                        how="outer")

    def visualize_model(self, plot, model):
        """Visualize the data for a mdl in a certain plot."""

        visualizer = plot(model)
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        visualizer.show()

    def evaluate_model(self, model, viz=False):
        """Run ML mdl and return score"""
        models = {"LR": linear_model.LinearRegression,
                  "DT": DecisionTreeClassifier,
                  "RF": RandomForestClassifier,
                  "RI": linear_model.Ridge,
                  "LA": linear_model.Lasso,
                  "EN": linear_model.ElasticNet}

        ml_model = models[model]()
        ml_model.fit(self.X_train, self.y_train)

        print(f"\n-----{str(model)}-----\n\n")

        if viz:
            # Create residuals plot
            for plot in [ResidualsPlot, PredictionError]:
                self.visualize_model(plot, ml_model)

        predictions = ml_model.predict(self.X_test)
        acc = median_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(ml_model.score(self.X_train, self.y_train))

        print(f"Model achieved an mean absolute error of {acc:.3f}."
              f"\nR2 score is {r2:.3f}")


if __name__ == '__main__':
    """Run script if directly loaded."""

    # prompt = "Name of file: "
    # validate_input(prompt, type_=str, min_=5)
    ML_mdl = MachineLearnModel("combination.pkl")
    mdls = ["LR"
        #, "DT", "RF"
            ]
    for mdl in mdls:
        ML_mdl.evaluate_model(mdl, viz=False)
    print("\nfinished")
