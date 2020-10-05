# (c) 2020 Rinze Douma

import importlib
# Import libraries
import os
from getpass import getpass

import geopandas as gpd
import numpy as np
import pandas as pd

importlib.import_module("helpers")
from helpers import convert_elapsed_time, extract_num, build_era    # noqa
from helpers import listing_type, roof_description                  # noqa
from helpers import garden, validate_input, contains_to_binary      # noqa

# Globals
BASE = os.path.join(os.pardir, "data")


def clean_dataset(filename):
    """Open, clean and save a Funda dataset."""

    # Open JSON file
    try:
        df = pd.read_json(os.path.join(BASE, filename))
    except FileNotFoundError:
        print("File not found.")
        return

    # Rename columns
    df = rename_cols(df)
    # Drop irrelevant cols & rows
    df = initial_drop(df)
    # Extract numeric data
    df = convert_num_cols(df)
    # Process binary information
    df = binary_columns(df)
    # Convert categorical data to dummy columns
    df = dummy_columns(df)

    # Drop columns that are not significant
    df.drop(columns=[
        'specials', 'bathroom_features', 'features',
        'isolation', 'heating', 'hot_water', 'boiler',
        'environment', 'garden_front', 'terrace', 'garden_back',
        'garden_patio', 'garage_type', 'storage_features', 'storage_isolation',
        'outdoor_m2', 'surface_m2', 'ownership'
    ], inplace=True)

    # Get coordinates and bin into neighborhoods
    print("Please provide Google Maps API key:")
    df = geolocation(df, getpass())

    # Export to pickle file
    path = os.path.join(BASE, filename.rsplit(".")[0] + ".pkl")
    df.to_pickle(path)
    print(f"Successfully exported to '{path}'.")


def rename_cols(df):
    """Translate and reorder columns."""

    print("Renaming columns...")

    # Set categories order
    col_trans = ["OVE", "BOU", "OPP", "OTH", "IND", "ENE", "BUI", "GAR",
                 "BER", "PAR", "VVE", "KAD", "BED", "VEI"]

    # Build list of column names incl category label
    order = [x
             for i, col in enumerate(col_trans)
             for x in df.columns
             if x.startswith(col_trans[i])]

    # Execute order
    df = df[["address", "postcode", "city"] + order].copy()

    # Translate column names
    df.rename(columns=TRANSLATE_COLS, inplace=True)

    return df


def initial_drop(df):
    """Drop data which is not relevant."""

    print("Dropping irrelevant cols & rows...")

    # Make sure the columns we use are also in this dataset
    add = [col for col in TRANSLATE_COLS.values() if col not in df.columns]
    df = pd.concat([df, pd.DataFrame(columns=add)], axis=1)

    # Drop extra columns present in this dataframe
    address = ["address", "postcode", "city"]
    extra = [col
             for col in df.columns
             if col not in list(TRANSLATE_COLS.values()) + address]

    # Initial drop of columns with little meaning
    drop = ['status', 'acceptance', 'asking_price_original', 'rent_price',
            'rental_agreement', 'rent_price_original', 'sale_type',
            'certificates', 'accessibility', 'prop_extra_type', 'parking_type',
            'prop_build_area', 'opp', 'parking_capacity',
            'prop_extra_dimensions', 'ground_area', 'garden_plaats',
            'garden_side', 'garage_size', 'garage_features', 'garage_isolation',
            'kadaster', 'comp_practice', 'company_space', 'office_space',
            'store_space', 'auction_price', 'auction_period', 'auction_type',
            'auction_party', "external_storage", "other_space",
            "ownership_costs"]
    df.drop(columns=drop+extra, inplace=True)

    # Drop rows without asking price
    df.dropna(subset=["asking_price"], inplace=True)

    # Drop listings that are just garages and such
    df = (df.drop(df[df["apartment_type"].isna()
                     & df["property_type"].isna()].index)
          .reset_index(drop=True))

    # Drop new build projects which aren't specific
    df.drop(
        df[
            df["address"].str.contains(r"bouw|appartemnt|wonen",
                                       case=False,
                                       regex=True)
        ].index,
        inplace=True)

    return df


def convert_num_cols(df):
    """Extract numeric values from columns."""

    print("Extracting numeric information...")

    # Columns in euro
    euro = ["asking_price", "vve_contribution",
            "service_fees_pm", "price_m2"]
    for e in euro:
        df[e] = extract_num(df[e], "price")

    # Calculate days since posting
    df["days_online"] = (df["days_online"]
                         .apply(convert_elapsed_time))

    # Calculate mean of build period
    df["build_era"] = (df["build_era"]
                       .apply(build_era)
                       .astype(int, errors="ignore"))

    # Use mean of build period if build year is null
    df["build_year"] = np.where(df["build_year"].notnull(),
                                df["build_year"],
                                df["build_era"])
    df.drop(columns=["build_era"], inplace=True)

    # Build year, bathrooms, toilets, area and volume
    df = (df
          .assign(build_year=extract_num(df["build_year"], "year"),
                  num_bathrooms=extract_num(df["bathrooms"], "bathrooms"),
                  num_toilets=extract_num(df["bathrooms"], "toilets"),
                  living_m2=extract_num(df["living_m2"], "meter"),
                  land_m2=extract_num(df["land_m2"], "meter"),
                  property_m3=extract_num(df["property_m3"], "meter"))
          .drop(columns="bathrooms"))

    # Extract number of rooms and bedrooms
    df["rooms"] = extract_num(df["num_rooms"], "rooms")
    df["bedrooms"] = extract_num(df["num_rooms"], "bedrooms")

    # Where no bedrooms mentioned, use rooms-1
    df.loc[(df["rooms"] > 1)
           & (df["bedrooms"] == 0), "bedrooms"] = df["rooms"] - 1

    # When rooms & bedrooms are both 2, increase rooms by 1
    df.loc[(df["rooms"] == 2) & (df["bedrooms"] == 2), "rooms"] = 3

    # For equal number use bedrooms-1 (if not already 1 bedroom)
    df.loc[(df["rooms"] == df["bedrooms"])
           & (df["rooms"] != 1), "bedrooms"] = df["bedrooms"] - 1

    # Drop original rooms column
    df.drop(columns="num_rooms", inplace=True)

    return df


def binary_columns(df):
    """When columns have essentially 2 values, set as either 1 or 0."""

    print("Setting binary columns...")

    # VVE columns
    vve = ["vve_per_contr", "vve_kvk", "vve_am", "vve_reserve_fund",
           "vve_maintenance", "vve_insurance"]

    # Fill NaN with 0
    df[vve] = df[vve].fillna(0)
    for col in vve:
        df[col] = contains_to_binary(df[col], "ja")

    # Other binary oppositions
    df["new_build"] = contains_to_binary(df["new_build"], "nieuwbouw")
    df["ownership"] = contains_to_binary(df["ownership"], "erfpacht")

    # Consider subtypes equal and set each category to 1 if available
    other_binary = ["balcony", "garden", "storage_type"]
    for col in other_binary:
        df[col] = np.where(df[col].notnull(), 1, 0)

    # Any on-street parking is considered 0
    pattern = r"openbaar|betaald|vergunning"
    df["parking"] = contains_to_binary(df["parking"], pattern, regex=True)

    return df


def dummy_columns(df):
    """Create dummy columns for categorical columns."""

    print("Create dummy columns for categorical columns...")

    # Property and apartment types (prefix pt)
    listing_type(df)

    # Dummies for roof 'type' and 'form' (prefix rt and rf)
    df = (pd.concat([df, roof_description(df["roof_type"].copy())],
                    axis=1)
          .drop(columns=["roof_type"]))

    # Attic and cellar
    pats = {"attic": "zolder|vliering", "cellar": "kelder"}
    for key, pat in pats.items():
        df["xf_" + key] = contains_to_binary(df["floors"], pat)

    # Since we have split the listing type into categories,
    # we can split for either apartments or full houses
    apartments = ["pt_bovenwoning", "pt_benedenwoning", "pt_penthouse",
                  "pt_corridorflat", "pt_portiekwoning"]
    other_prop = [col
                  for col in df.columns
                  if col.startswith("pt") and col not in apartments]

    # Number of floors in house
    df["floors"] = np.where(df[other_prop].apply(any, axis=1),
                            extract_num(df["floors"], "floors"),
                            0)

    # Level of apartment
    df["apartment_level"] = extract_num(df["apartment_level"], "app_level")

    # Set energy label, fall back to temp label and level G
    df["energy_label"] = np.where(
        df["energy_label"].str.len() == 1,
        df["energy_label"],
        np.where(
            df["energy_label_temp"].str.len() == 1,
            df["energy_label_temp"],
            "G")
    )

    # Dummies for energy
    df = pd.get_dummies(df,
                        columns=["energy_label"],
                        drop_first=True,
                        prefix="en")

    df.drop(columns=["energy_label_temp"], inplace=True)

    # Garden orientation
    df["garden_orientation"] = df["garden_orientation"].apply(garden)
    df = pd.get_dummies(df,
                        columns=["garden_orientation"],
                        drop_first=True,
                        prefix="ga")

    return df


def geolocation(df, key):
    """Bin listings into neighborhoods."""

    print("Fetch coordinates and bin into neighborhoods...")

    # Delete additions to house number and drop rows without house number
    df["address"] = df["address"].str.extract(r"(.+ \d+)", expand=False)
    df.dropna(subset=["address"], inplace=True)

    # Make the full address into one column
    df = (df.assign(full_address=df[["address", "postcode", "city"]]
                    .agg(", ".join, axis=1))
          .drop(columns=["address", "postcode", "city"]))

    # Rename address column and bring to front
    df.insert(0, column="address", value=df.pop("full_address"))

    # Get the coordinates of every location
    coords = (gpd.tools.geocode(df["address"],
                                provider="google",
                                api_key=key))

    # Make sure downloaded coordinates are in correct format
    coords.to_crs(epsg=4326, inplace=True)

    # Load neighborhoods of Amsterdam
    gdf = gpd.read_file(os.path.join(BASE, "geo", "ams_neighborhoods.shp"))

    # Add the neighborhood for every location
    geo_combi = (gpd
                 .sjoin(coords,
                        gdf[["Buurt", "geometry"]],
                        how="inner",
                        op='intersects')
                 .drop(columns=["geometry", "index_right"])
                 .sort_index())
    geo_combi.head()

    # Merge with original dataframe
    df = (df.merge(geo_combi,
                   left_index=True,
                   right_index=True)
          .drop(columns=["address_y"])
          .rename(columns={"address_x": "address"}))

    # Clean neighborhood names
    df["Buurt"] = (df["Buurt"]
                   .str.replace(" ", "_")
                   .str.replace(".", "")
                   .str.lower())

    # Get dummies for neighborhood
    df = pd.get_dummies(df,
                        columns=["Buurt"],
                        prefix="ne")

    return df


def combine_df(f1, f2, output=None):
    """Combine 2 dataframes in one and save as new pickle."""

    # Open files
    dataframes = []
    for filename in [f1, f2]:
        dataframes = pd.read_pickle(os.path.join(BASE, filename))

    # Combine
    df = pd.concat(dataframes).fillna(0).reset_index(drop=True)

    # Export
    if not output:
        output = "combination.pkl"
    df.to_pickle(os.path.join(BASE, output))


# Translate column names
TRANSLATE_COLS = {'OVE-Vraagprijs': 'asking_price',
                  'OVE-Vraagprijs per m²': 'price_m2',
                  'OVE-Aangeboden sinds': 'days_online',
                  'OVE-Status': 'status',
                  'OVE-Aanvaarding': 'acceptance',
                  'BOU-Soort woonhuis': 'property_type',
                  'BOU-Soort bouw': 'new_build',
                  'BOU-Bouwjaar': 'build_year',
                  'BOU-Soort dak': 'roof_type',
                  'OPP-': 'opp',
                  'OPP-Perceel': 'land_m2',
                  'OPP-Inhoud': 'property_m3',
                  'IND-Aantal kamers': 'num_rooms',
                  'IND-Aantal badkamers': 'bathrooms',
                  'IND-Badkamervoorzieningen': 'bathroom_features',
                  'IND-Aantal woonlagen': 'floors',
                  'IND-Voorzieningen': 'features',
                  'ENE-Energielabel': 'energy_label',
                  'ENE-Isolatie': 'isolation',
                  'ENE-Verwarming': 'heating',
                  'ENE-Warm water': 'hot_water',
                  'ENE-Cv-ketel': 'boiler',
                  'KAD-': 'kadaster',
                  'BUI-Tuin': 'garden',
                  'BUI-Balkon/dakterras': 'balcony',
                  'GAR-Soort garage': 'garage_type',
                  'GAR-Capaciteit': 'garage_size',
                  'GAR-Voorzieningen': 'garage_features',
                  'PAR-Soort parkeergelegenheid': 'parking',
                  'BUI-Ligging': 'environment',
                  'BUI-Voortuin': 'garden_front',
                  'BUI-Ligging tuin': 'garden_orientation',
                  'OVE-Servicekosten': 'service_fees_pm',
                  'BOU-Specifiek': 'specials',
                  'BUI-Zonneterras': 'terrace',
                  'BER-Schuur/berging': 'storage_type',
                  'BER-Voorzieningen': 'storage_features',
                  'BUI-Achtertuin': 'garden_back',
                  'BER-Isolatie': 'storage_isolation',
                  'BOU-Keurmerken': 'certificates',
                  'BUI-Patio/atrium': 'garden_patio',
                  'OVE-Bijdrage VvE': 'vve_contribution',
                  'BOU-Soort appartement': 'apartment_type',
                  'BOU-Bouwperiode': 'build_era',
                  'IND-Gelegen op': 'apartment_level',
                  'VVE-Inschrijving KvK': 'vve_kvk',
                  'VVE-Jaarlijkse vergadering': 'vve_am',
                  'VVE-Periodieke bijdrage': 'vve_per_contr',
                  'VVE-Reservefonds aanwezig': 'vve_reserve_fund',
                  'VVE-Onderhoudsplan': 'vve_maintenance',
                  'VVE-Opstalverzekering': 'vve_insurance',
                  'GAR-Isolatie': 'garage_isolation',
                  'BOU-Toegankelijkheid': 'accessibility',
                  'OVE-Oorspronkelijke vraagprijs': 'asking_price_original',
                  'ENE-Voorlopig energielabel': 'energy_label_temp',
                  'OVE-Huurprijs': 'rent_price',
                  'OVE-Huurovereenkomst': 'rental_agreement',
                  'BUI-Plaats': 'garden_plaats',
                  'BUI-Zijtuin': 'garden_side',
                  'OVE-Oorspronkelijke huurprijs': 'rent_price_original',
                  'BOU-Soort overig aanbod': 'prop_extra_type',
                  'BED-Praktijkruimte': 'comp_practice',
                  'OVE-Koopmengvorm': 'sale_type',
                  'BOU-Soort parkeergelegenheid': 'parking_type',
                  'IND-Capaciteit': 'parking_capacity',
                  'IND-Afmetingen': 'prop_extra_dimensions',
                  'VEI-Prijs': 'auction_price',
                  'VEI-Veilingperiode': 'auction_period',
                  'VEI-Soort veiling': 'auction_type',
                  'VEI-Veilingpartij': 'auction_party',
                  'BED-Bedrijfsruimte': 'company_space',
                  'BED-Kantoorruimte': 'office_space',
                  'BED-Winkelruimte': 'store_space',
                  'IND-Perceel': 'ground_area',
                  'BOU-Soort object': 'prop_build_area',
                  'OTH-Wonen': 'living_m2',
                  'OTH-Gebouwgebonden buitenruimte': 'outdoor_m2',
                  'OTH-Oppervlakte': 'surface_m2',
                  'OTH-Eigendomssituatie': 'ownership',
                  'OVE-Laatste vraagprijs': 'asking_price',
                  'OTH-Externe bergruimte': "external_storage",
                  'OTH-Overige inpandige ruimte': "other_space",
                  'OTH-Lasten': "ownership_costs"}

if __name__ == '__main__':
    prompt = "Please provide filename of .json in data folder: "
    f_name = validate_input(prompt, type_=str, min_=5)
    clean_dataset(f_name)
