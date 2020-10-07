# (c) 2020 Rinze Douma

import pandas as pd
import numpy as np


def convert_elapsed_time(x):
    """Return integer of days since listing online."""

    translation = {"januari": "january",
                   "februari": "february",
                   "maart": "march",
                   "april": "april",
                   "mei": "may",
                   "juni": "june",
                   "juli": "july",
                   "augustus": "august",
                   "september": "september",
                   "oktober": "october",
                   "november": "november",
                   "december": "december"}

    try:
        parts = x.split()
    except AttributeError:
        return x

    if len(parts) == 3:
        x = "-" .join([parts[0], translation[parts[1].lower()], parts[2]])
        days = (pd.to_datetime("now") - pd.to_datetime(x)).days
    elif len(parts) == 2:
        time_units = {"weken": 7, "maanden": 40}
        days = int(parts[0].strip("+")) * time_units[parts[1].lower()]
    else:
        days = 1

    return int(days)


def extract_num(col, mode):
    """Return column with numeric of digits in column."""

    pat = {"price": r"€ (\S+)",
           "meter": r"(\d+)(?: m)",
           "year": r"(\d+)",
           "rooms": r"(\d+)(?: kamer)",
           "bedrooms": r"(\d+)(?: slaapkamer)",
           "floors": r"(\d+)(?: woonla)",
           "bathrooms": r"(\d+)(?: badk)",
           "toilets": r"(\d+)(?: apart)",
           "app_level": r"(\d+)(?:e woonl)"}

    return pd.to_numeric(col
                         .astype(str)
                         .str.extract(pat[mode], expand=False)
                         .str.replace(".", "")
                         .str.replace(",", ".")
                         .fillna(0))


def contains_to_binary(col, pattern, regex=False, opp=False):
    """Return 1 if col contains pattern, else 0."""

    # Set values for true and false conditions based on func argument
    t, f = (0, 1) if opp else (1, 0)

    return np.where(col
                    .astype(str)
                    .str.contains(pattern,
                                  case=False,
                                  regex=regex,
                                  na=False),
                    t, f)


def build_era(x):
    """Return mean of build time period."""

    if x != x:
        return pd.NA
    start, end = x.split("-")

    return int((int(start) + int(end)) / 2)


def listing_type(df):
    """Split tags in property and apartment columns to dummies."""

    # Merge houses and APARTMENTS to one column
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
    all_tags = {tag
                for c in tags.columns
                for tag in tags[c].unique()
                if tag and tag == tag}

    # Concatenate with original DF
    pd.concat([df, pd.DataFrame(columns=[tag for tag in all_tags])], axis=1)

    # Evaluate whether tag applicable
    for tag in all_tags:
        df[tag] = contains_to_binary(df["property_type"], tag)

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
        elem = [e for e in [key]+elem if e in all_tags]

        # If any of the key+elem is 1, set the key to 1
        df[key] = np.where(df[elem].apply(any, axis=1), 1, 0)

    # Drop the columns we have combined into others
    useless = ["service flat",
               "bedrijfs- of dienstwoning",
               "appartement met open portiek",
               "appartement"]
    drop = [col
            for lst in useless+list(combine.values())
            for col in lst
            if col in all_tags]
    old_cols = ["apartment_type", "property_type"]
    df.drop(columns=drop+old_cols, inplace=True)

    # Add a prefix and replace spaces
    df.columns = [f"pt_{col}".replace(" ", "_")
                  if col in all_tags
                  else col
                  for col in df.columns]


def roof_description(col):
    """Return DataFrame with dummy columns for roof type and form."""

    roof_form = ["zadeldak", "plat dak", "lessenaardak", "mansarde",
                 "samengesteld", "tentdak", "dwarskap", "schilddak"]
    roof_type = ["bitumin", "pannen", "kunststof", "leisteen",
                 "metaal", "asbest", "riet"]

    # Create dataframe which we will return later
    dummies = pd.DataFrame(index=col.index)

    # Create dummy column for each value
    for pat in roof_form:
        dummies[f"rf_{pat}"] = contains_to_binary(col, pat)
        dummies.rename(columns={"rf_plat dak": "rf_plat_dak"})
    for pat in roof_type:
        dummies[f"rt_{pat}"] = contains_to_binary(col, pat)

    return dummies


def garden(x):
    """Return only cardinal wind directions."""

    orientations = ["noord", "zuid", "west", "oost"]

    for direction in orientations:
        try:
            if direction in x:
                return direction
        except TypeError:
            return pd.NA
    return pd.NA


def validate_input(prompt, type_=None, min_=None, max_=None, options=None):
    """ Request user input and clean it before return.
    :param prompt: Question to ask user.
    :param type_: Type of value asked. str, int, float.
    :param min_: Minimum length of str of lower value of int.
    :param max_: Maximum length of str of upper value of int.
    :param options: List of options allowed
    :return: str, int or float.

    adapted from https://stackoverflow.com/questions/23294658/
        asking-the-user-for-input-until-they-give-a-valid-response
    """

    if (min_ is not None
            and max_ is not None
            and max_ < min_):
        raise ValueError("min_ must be less than or equal to max_.")
    while True:
        ui = input(prompt)
        if type_ is not None:
            try:
                ui = type_(ui)
            except ValueError:
                print(f"Input type must be {type_.__name__}")
                continue
        if isinstance(ui, str):
            ui_num = len(ui)
        else:
            ui_num = ui
        if max_ is not None and ui_num > max_:
            print(f"Input must be less than or equal to {max_}.")
        elif min_ is not None and ui_num < min_:
            print(f"Input must be more than or equal to {min_}.")
        elif options is not None and ui.lower() not in options:
            print("Input must be one of the following: " + ", ".join(options))
        else:
            return ui


def log_print(msg, verbose):
    """Helper function to print log messages."""
    if verbose:
        print(msg)
