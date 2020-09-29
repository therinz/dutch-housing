import pandas as pd
import numpy as np


def convert_elapsed_time(x):
    """Return integer of days since listing online."""

    parts = x.split()
    if len(parts) == 3:
        days = (pd.to_datetime("now") - pd.to_datetime(x)).days
    elif len(parts) == 2:
        time_units = {"weken": 7, "maanden": 40}
        days = int(parts[0].strip("+")) * time_units[parts[1].lower()]
    else:
        days = 1

    return int(days)


def extract_num(x, mode):
    """Return column with integer of number in column."""

    pat = {"price": r"€ (\S+)",
           "meter": r"(\d+)(?: m)",
           "year": r"(\d+)"}

    return pd.to_numeric(x.astype(str)
                         .str.extract(pat[mode], expand=False)
                         .str.replace(".", "")
                         .str.replace(",", ".")
                         .fillna(0))


def build_era(x):
    """Return mean of build time period."""

    if x != x:
        return pd.NA
    start, end = x.split("-")

    return int((int(start) + int(end)) / 2)


def add_prefix(s, prefix):
    """Add a prefix to distinguish dummies"""
    return f"{prefix}_{s}"


def listing_type(df):
    """Split tags in property and apartment columns to dummies."""

    # Merge houses and apartments to one column
    df["property_type"] = np.where(df["property_type"].notna(),
                                   df["property_type"],
                                   df["apartment_type"])
    df.drop(columns=["apartment_type"], inplace=True)

    # Flatten parentheses to a comma separated value and split on comma
    tags = (df["property_type"]
            .str.replace(" \(", ", ")
            .str.replace("\)", "")
            .str.lower()
            .str.split(", ", expand=True))

    # Save all tags in a set
    all_tags = {t
                for c in tags.columns
                for t in tags[c].unique()
                if t and t == t}

    # Concatenate with original DF
    pd.concat([df, pd.DataFrame(columns=[add_prefix(t, "pt")
                                         for t in all_tags])], axis=1)

    # Evaluate whether tag applicable
    for tag in all_tags:
        df[add_prefix(tag, "pt")] = np.where(df["property_type"]
                                             .str.contains(tag, case=False),
                                             1, 0)

    # Cleanup column names
    df.columns = df.columns.str.replace(" ", "_")

    # Many tags mean the same, so we combine them under a single header
    combine = {'2-onder-1-kapwoning': ['geschakelde_2-onder-1-kapwoning',
                                       'halfvrijstaande_woning'],
               'appartement': ['appartement_met_open_portiek',
                               'dubbel_bovenhuis',
                               'dubbel_bovenhuis_met_open_portiek',
                               'maisonnette', 'tussenverdieping',
                               'bovenwoning', 'beneden_+_bovenwoning'],
               'hoekwoning': ['eindwoning'],
               'waterwoning': ['woonboot'],
               'portiekwoning': ['open_portiek', 'portiekflat'],
               'landhuis': ['landgoed', 'woonboerderij'],
               'bungalow': ['semi-bungalow'],
               'eengezinswoning': ['patiowoning', 'dijkwoning',
                                   'split-level_woning', 'kwadrant_woning'],
               'benedenwoning': ['dubbel_benedenhuis', 'bel-etage',
                                 'souterrain'],
               'herenhuis': ['grachtenpand'],
               'corridorflat': ['galerijflat'],
               'villa': ['vrijstaande_woning'],
               'tussenwoning': ['geschakelde_woning']}

    for key, elem in combine.items():
        key = add_prefix(key, "pt")
        elem = [add_prefix(e, "pt") for e in elem]
        df[key] = np.where(df[[key] + elem].apply(any, axis=1), 1, 0)

    # Drop the columns we have combined into others
    drop = [add_prefix(col, "pt")
            for lst in combine.values()
            for col in lst]
    useless = [add_prefix("service_flat", "pt"),
               add_prefix("bedrijfs-_of_dienstwoning", "pt")]
    df.drop(columns=drop+useless, inplace=True)


def roof_description(col):
    """Return DataFrame with dummy columns for roof type and form."""

    roof_form = ["zadeldak", "plat dak", "lessenaardak",
                 "mansarde", "samengesteld", "tentdak",
                 "dwarskap", "schilddak"]
    roof_type = ["bitumin", "pannen", "kunststof", "leisteen",
                 "metaal", "asbest", "riet"]

    # Create dataframe we will return later
    dummies = pd.DataFrame()

    # Create dummies
    for pat in roof_form:
        dummies[add_prefix(pat, "rf")] = np.where(col.str.contains(pat,
                                                                   case=False,
                                                                   na=False),
                                                  1, 0)
    for pat in roof_type:
        dummies[add_prefix(pat, "rt")] = np.where(col.str.contains(pat,
                                                                   case=False,
                                                                   na=False),
                                                  1, 0)

    return dummies

test = 3