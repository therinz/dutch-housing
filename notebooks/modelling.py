# (c) 2020 Rinze Douma

import os
import importlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, median_absolute_error
from sklearn import linear_model
from yellowbrick.regressor import ResidualsPlot, PredictionError

importlib.import_module("helpers")
from helpers import validate_input  # noqa

importlib.import_module("json_dataframe")
from json_dataframe import APARTMENTS, clean_dataset  # noqa


class MachineLearnModel:
    base = os.path.join(os.pardir, "data")

    def __init__(self, filename, apartment=False):
        # Declare variables
        self.X_train = self.X_test = self.y_train = self.y_test = pd.DataFrame
        self.q = pd.DataFrame
        self.scaled_fit = self.ml_model = None

        # Open file
        self.df = pd.read_pickle(os.path.join(MachineLearnModel.base,
                                              filename))

        # Check for null values
        if any(self.df.isnull().any()):
            raise ValueError("Null values present in DataFrame.")

        # Temp solution for incorrect column names
        self.df = (self.df.rename(columns={"rf_plat dak": "rf_plat_dak"})
                   .drop(columns=["xf_attic"])
                   .reset_index(drop=True))

        # Drop columns that appear highly correlated with other factors.
        distractions = ["vve_kvk", "vve_am", "vve_per_contr",
                        "vve_reserve_fund", "rt_pannen", "rf_plat_dak",
                        "vve_contribution", "address", "price_m2"]
        self.df = self.df.drop(columns=distractions)

        # Select apartments or houses
        if apartment:
            self.df = self.df[self.df[APARTMENTS].apply(any, axis=1)]
            self.apartments()
        else:
            self.df = self.df[~self.df[APARTMENTS].apply(any, axis=1)]
            self.houses()

        # Split into X and y
        self.split_dataset()

        # Scale X
        self.scaler()

    def apartments(self):
        """Remove outliers and drop non-apartment columns."""

        outliers = {
            *self.df[(self.df["asking_price"] > 10000000)
                      | (self.df["asking_price"] < 100000)].index,
            *self.df[self.df["build_year"] < 1600].index,
            *self.df[self.df["service_fees_pm"] > 500].index,
            *self.df[(self.df["property_m3"] < 10)
                     | (self.df["property_m3"] > 800)].index,
            *self.df[self.df["num_bathrooms"] > 4].index,
            *self.df[self.df["num_toilets"] > 3].index,
            *self.df[self.df["bedrooms"] > 5].index,
            *self.df[self.df["days_online"] > 120].index
        }

        # Drop rows that have outliers
        self.df = self.df.drop(outliers)

        # Drop columns that are not relevant for apartments
        pt = [col
              for col in self.df.columns
              if (col.startswith("pt") and col not in APARTMENTS)
              or (col.startswith("rt_") or col.startswith("rf_"))]
        self.df = self.df.drop(columns=pt + ["land_m2", "floors"])

    def houses(self):
        """Remove outliers and drop apartment columns."""

        outliers = {
            *self.df[(self.df["asking_price"] > 10000000)
                         | (self.df["asking_price"] < 100000)].index,
            *self.df[self.df["build_year"] < 1600].index,
            *self.df[self.df["land_m2"] > 600].index,
            *self.df[(self.df["property_m3"] < 10)
                         | (self.df["property_m3"] > 800)].index,
            *self.df[self.df["living_m2"] > 500].index,
            *self.df[self.df["num_bathrooms"] > 5].index,
            *self.df[self.df["num_toilets"] > 4].index,
            *self.df[self.df["floors"] > 7].index,
            *self.df[self.df["rooms"] > 15].index,
            *self.df[self.df["bedrooms"] > 8].index,
            *self.df[self.df["days_online"] > 120].index
        }

        # Drop rows that have outliers
        self.df = self.df.drop(outliers)

        # Drop columns that are not relevant for houses
        pt = [col
              for col in self.df.columns
              if col in APARTMENTS]
        self.df = self.df.drop(columns=pt + ["apartment_level"])

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

        # In prediction apartment: only fit and return
        if self.scaled_fit:
            self.q.reset_index(drop=True, inplace=True)
            scaled = pd.DataFrame(self.scaled_fit
                                  .transform(self.q[num_cols]),
                                  columns=num_cols)
            self.q = self.q.drop(columns=num_cols, axis=1)
            self.q = self.q.merge(scaled,
                                  left_index=True,
                                  right_index=True,
                                  how="outer")
            return

        # Fit scaler model
        std = StandardScaler()
        self.scaled_fit = std.fit(self.X_train[num_cols])

        # Apply to dataframe, train set first
        self.X_train.reset_index(drop=True, inplace=True)
        scaled = pd.DataFrame(self.scaled_fit
                              .transform(self.X_train[num_cols]),
                              columns=num_cols)
        self.X_train = self.X_train.drop(columns=num_cols, axis=1)
        self.X_train = self.X_train.merge(scaled,
                                          left_index=True,
                                          right_index=True,
                                          how="outer")
        # test set
        self.X_test.reset_index(drop=True, inplace=True)
        scaled_test = pd.DataFrame(self.scaled_fit
                                   .transform(self.X_test[num_cols]),
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

    def evaluate_model(self, model, viz=False, save=False, verbose=True):
        """Run ML mdl and return score"""
        models = {"LR": linear_model.LinearRegression,
                  "RI": linear_model.Ridge,
                  "LA": linear_model.Lasso,
                  "EN": linear_model.ElasticNet}

        trans = {"LR": "Linear regression model",
                 "RI": "Ridge regression model",
                 "LA": "Lasso regression model",
                 "EN": "ElasticNet regression model"}

        ml_model = models[model]()
        ml_model.fit(self.X_train, self.y_train)

        print(f"\n-----Working on: {str(trans[model])}...-----")

        if viz:
            # Create residuals plot
            for plot in [ResidualsPlot, PredictionError]:
                self.visualize_model(plot, ml_model)

        # Print stats
        if verbose:
            # Get predictions and score it
            predictions = ml_model.predict(self.X_test)
            train_r2 = ml_model.score(self.X_train, self.y_train)
            acc = median_absolute_error(self.y_test, predictions)
            test_r2 = r2_score(self.y_test, predictions)

            # Print to screen
            size = self.X_train.shape[0] + self.X_test.shape[0]
            print(f"Total rows used: {size}"
                  f"\nR2 for training set: {train_r2:.3f}."
                  f"\nMean absolute error of {int(acc)}."
                  f"\nR2 score for test set: {test_r2:.3f}\n---")

        # Save the model for prediction later
        if save:
            self.ml_model = ml_model

    def predict(self, file):
        """Predict price based on characteristics."""

        print("\nFetching new data:")

        # Open file and remove asking_price
        self.q = (clean_dataset(file, mode="predict")
                  .drop(columns=["asking_price"]))
        address = self.q.iloc[0]["address"]

        # Equalize columns with X_train
        add = set(self.X_train.columns).difference(set(self.q.columns))
        remove = set(self.q.columns).difference(set(self.X_train.columns))
        add_cols = dict.fromkeys(add, 0)
        self.q = self.q.drop(columns=remove).assign(**add_cols)

        # Scale q
        self.scaler()

        # Predict
        value = int(abs(self.ml_model.predict(self.q)))
        print(f"\n-----\nExpected asking price for {address}: € {value},-.")


def lookup_worth():
    # Ask for type of lookup
    prompt = "Type of listing: apartment or house? "
    options = ["apartment", "a", "house", "h"]
    question = validate_input(prompt, type_=str, options=options)

    # Train model
    mode = question in options[:2]
    ML_mdl = MachineLearnModel("combination.pkl", apartment=mode)
    ML_mdl.evaluate_model("EN", viz=False, save=True, verbose=False)

    # Make prediction
    ML_mdl.predict("predict.json")


if __name__ == '__main__':
    """Run script if directly loaded."""

    # prompt = "Name of file: "
    # validate_input(prompt, type_=str, min_=5)

    # Initialize model
    ML_mdl = MachineLearnModel("combination.pkl", apartment=True)
    mdls = [#"LR", "RI", "LA",
        "EN"
            ]
    for mdl in mdls:
        ML_mdl.evaluate_model(mdl, viz=False, save=True)
    print("Finished shaping model.")
    ML_mdl.predict("predict.json")
