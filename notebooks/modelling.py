
import os

import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

BASE = os.path.join(os.pardir, "data")

df = pd.read_pickle(os.path.join(BASE, "intermediate.pkl"))

# Set variables
X = df[x_cols]
y = df["asking_price"]
# Instantiate the class
lin_model = linear_model.LinearRegression()
# Create the model
lin_model.fit(X, y)


def scaler(df):
    """Convert numeric columns into scaled values."""

    # Select column names of factors with more than 2 values
    num_cols = [col for col in df.columns
                if df[col].nunique() > 2
                and df[col].dtype in ["int64", "float64"]]

    # Fit model
    std = StandardScaler()
    scaled_fit = std.fit(df[num_cols])

    # Apply
    df[num_cols] = pd.DataFrame(scaled_fit.transform(df[num_cols]),
                                columns=num_cols)

    return df


def visualize_model(plot, model, train, test):
    """Visualize the data for a model in a certain plot."""

    visualizer = plot(model)
    visualizer.fit(*train)
    visualizer.score(*test)
    visualizer.show()


