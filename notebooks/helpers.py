# (c) 2020 Rinze Douma

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


def extract_num(col, mode):
    """Return column with integer of number in column."""

    pat = {"price": r"€ (\S+)",
           "meter": r"(\d+)(?: m)",
           "year": r"(\d+)",
           "rooms": r"(\d+)(?: kamer)",
           "bedrooms": r"(\d+)(?: slaapkamer)",
           "floors": r"(\d+)(?: woonla)",
           "bathrooms": r"(\d+)(?: badk)",
           "toilets": r"(\d+)(?: apart)",
           "app_level": r"(\d+)(?:e woonl)"}

    return pd.to_numeric(col.astype(str)
                         .str.extract(pat[mode], expand=False)
                         .str.replace(".", "")
                         .str.replace(",", ".")
                         .fillna(0))


def create_dummy(col, pattern):
    """Return 1 if col contains pattern, else 0."""

    return np.where(col.str.contains(pattern, case=False, na=False), 1, 0)


def build_era(x):
    """Return mean of build time period."""

    if x != x:
        return pd.NA
    start, end = x.split("-")

    return int((int(start) + int(end)) / 2)


def pref(s, prefix):
    """Add a prefix to distinguish dummies"""
    return f"{prefix}_{s}"


def listing_type(df):
    """Split tags in property and apartment columns to dummies."""

    # Merge houses and apartments to one column
    df["property_type"] = np.where(df["property_type"].notna(),
                                   df["property_type"],
                                   df["apartment_type"])

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
    pd.concat([df, pd.DataFrame(columns=[pref(t, "pt")
                                         for t in all_tags])], axis=1)

    # Evaluate whether tag applicable
    for tag in all_tags:
        df[pref(tag, "pt")] = create_dummy(df["property_type"], tag)

    # Cleanup column names
    df.columns = df.columns.str.replace(" ", "_")

    # Many tags mean the same, so we combine them under a single header
    combine = {'2-onder-1-kapwoning': ['geschakelde_2-onder-1-kapwoning',
                                       'halfvrijstaande_woning'],
               'bovenwoning': ['dubbel_bovenhuis',
                               'dubbel_bovenhuis_met_open_portiek',
                               'maisonnette', 'tussenverdieping',
                               'beneden_+_bovenwoning'],
               'benedenwoning': ['dubbel_benedenhuis', 'bel-etage',
                                 'souterrain'],
               'hoekwoning': ['eindwoning'],
               'waterwoning': ['woonboot'],
               'portiekwoning': ['open_portiek', 'portiekflat'],
               'landhuis': ['landgoed', 'woonboerderij'],
               'bungalow': ['semi-bungalow'],
               'eengezinswoning': ['patiowoning', 'dijkwoning',
                                   'split-level_woning', 'kwadrant_woning',
                                   'hofjeswoning'],
               'herenhuis': ['grachtenpand'],
               'corridorflat': ['galerijflat'],
               'villa': ['vrijstaande_woning'],
               'tussenwoning': ['geschakelde_woning']}

    for key, elem in combine.items():
        key = pref(key, "pt")
        elem = [pref(e, "pt") for e in elem]

        # If any of the key+elem is 1, set the key to 1
        df[key] = np.where(df[[key] + elem].apply(any, axis=1)
                           , 1, 0)

    # Drop the columns we have combined into others
    drop = [pref(col, "pt")
            for lst in combine.values()
            for col in lst]
    useless = [pref("service_flat", "pt"),
               pref("bedrijfs-_of_dienstwoning", "pt"),
               "pt_appartement_met_open_portiek",
               "pt_appartement",
               "apartment_type",
               "property_type"]
    df.drop(columns=drop+useless, inplace=True)


def roof_description(col):
    """Return DataFrame with dummy columns for roof type and form."""

    roof_form = ["zadeldak", "plat dak", "lessenaardak",
                 "mansarde", "samengesteld", "tentdak",
                 "dwarskap", "schilddak"]
    roof_type = ["bitumin", "pannen", "kunststof", "leisteen",
                 "metaal", "asbest", "riet"]

    # Create dataframe which we will return later
    dummies = pd.DataFrame(index=col.index)

    # Create dummy column for each value
    for pat in roof_form:
        dummies[pref(pat, "rf")] = create_dummy(col, pat)
    for pat in roof_type:
        dummies[pref(pat, "rt")] = create_dummy(col, pat)

    return dummies


def garden(x):
    orientations = ["noord", "zuid", "west", "oost"]

    for direction in orientations:
        try:
            if direction in x:
                return direction
        except TypeError:
            return pd.NA
    return pd.NA

