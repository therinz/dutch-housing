# (c) 2020 Rinze Douma

import os

import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


BASE = os.path.join(os.pardir, "data")


def modelling(filename):
    """Open a file and run various models."""

    # Open file
    df = pd.read_pickle(os.path.join(BASE, filename))

    # Temp solution for incorrect column names
    df.rename(columns={"rf_plat dak": "rf_plat_dak",
                       "address_x": "address"},
              inplace=True)
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Scale X
    X_train, X_test = scaler(X_train, X_test)

    models = [linear_model, DecisionTreeClassifier, RandomForestClassifier]
    for model in models:
        evaluate_model(X_train, X_test, y_train, y_test, model)


def split_dataset(xf, test_size=.4):
    """Return train & test set for X and y."""

    # Drop columns that appear highly correlated with other factors.
    distractors = ["vve_kvk", "vve_am", "vve_per_contr", "vve_reserve_fund",
                   "rt_pannen", "rf_plat_dak", "address", "price_m2"]

    # Set variables
    X = xf[[col
            for col in xf.columns
            if col not in ["asking_price"] + distractors]]
    y = xf["asking_price"]

    return tts(X, y, test_size=test_size, random_state=7)


def scaler(train, test):
    """Convert numeric columns into scaled values."""

    # Select column names of factors with more than 2 values
    num_cols = [col for col in train.columns
                if train[col].nunique() > 2
                and train[col].dtype in ["int64", "float64"]]

    # Fit model
    std = StandardScaler()
    scaled_fit = std.fit(train[num_cols])

    # Apply
    for xf in [train, test]:
        scaled_X = scaled_fit.transform(xf[num_cols])
        xf[num_cols] = pd.DataFrame(scaled_X, columns=num_cols)

    return train, test


def visualize_model(plot, model, train, test):
    """Visualize the data for a model in a certain plot."""

    visualizer = plot(model)
    visualizer.fit(*train)
    visualizer.score(*test)
    visualizer.show()


def evaluate_model(train_X, test_X, train_y, test_y, model):
    ml_model = model()
    ml_model.fit(train_X, train_y)

    predictions = ml_model.predict(test_X)
    acc = accuracy_score(test_y, predictions)

    print(f"{model} achieved an accuracy of {acc}")
