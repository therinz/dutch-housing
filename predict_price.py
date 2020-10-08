# (c) 2020 Rinze Douma

import os
import scrapy

import pandas as pd
from scraping.scraping.spiders.funda_spider import value_block
from scrapy.crawler import CrawlerProcess

from helpers import validate_input, log_print
from modelling import MachineLearnModel
from json_dataframe import APARTMENTS, clean_dataset


class PredictSpider(scrapy.Spider):
    name = "predict"
    start_urls = []
    allowed_domains = ["funda.nl"]
    custom_settings = {'DEPTH_LIMIT': 0}

    def __init__(self, link=None, *args, **kwargs):
        super(PredictSpider, self).__init__(*args, **kwargs)

        self.start_urls = [link]

    def parse(self, response):
        """Parse information found on house description page."""

        # Retrieve basic info about listing
        subtitle = response.css(".object-header__subtitle::text").get().split()
        info = {"address": response.css(".object-header__title::text").get(),
                "postcode": " ".join(subtitle[0:2]),
                "city": " ".join(subtitle[2:])}

        # Build list of information blocks
        dls = response.xpath("//dl")
        info.update({k: v
                     for dl in dls
                     for k, v in value_block(dl).items()})

        yield info


JSON = os.path.join("data", "predict.json")


def lookup_worth(verbose=False, debug=False):
    """Return predicted value of given listing."""

    # Debug mode to use stored data
    log_print("Debug: on.", verbose)
    if debug:
        df = pd.read_pickle(os.path.join("data", "predict.pkl"))
    else:
        # Cleanup before we start
        if os.path.exists(JSON):
            os.remove(JSON)

        # Ask for type of lookup
        prompt = "Whats the URL of the house? \n"
        url = ""
        while "funda.nl/" not in url:
            url = validate_input(prompt, type_=str, min_=10).strip(" \"\'")
            prompt = "Not a valid url! Please try again: \n"

        # Set settings for crawler
        settings = {"FEEDS": {JSON: {"format": "json"}},
                    "LOG_ENABLED": True}
        if verbose:
            print("Retrieving data from funda.nl...")
        else:
            print("\nHang on, doing some magic...")
            settings.update({"LOG_LEVEL": "WARNING"})

        # Let crawler work
        process = CrawlerProcess(settings)
        process.crawl(PredictSpider, link=url)
        process.start()

        # Clean the data
        df = clean_dataset(JSON, predict=True, verbose=verbose)

        # Save for future debugging
        df.to_pickle(os.path.join("data", "predict.pkl"))

    # Store real asking price and then delete column
    ap = df.iloc[0]["asking_price"]
    df = df.drop(columns=["asking_price"])

    # Set mode to apartments or houses
    apartments = [col for col in df.columns
                  if col.startswith("pt") and col in APARTMENTS]
    mode = df[apartments].apply(any, axis=1)[0]
    log_print("Mode: apartment" if mode else "Mode: house", verbose)

    # Train model
    ML_mdl = MachineLearnModel(apartment=mode, verbose=verbose)
    if debug:
        mdls = [
            # "RI",
            "LA",
            # "EN"
        ]
        for mdl in mdls:
            ML_mdl.evaluate_model(mdl, viz=False, save=True)
    else:
        mdl = "LA" if mode else "RI"
        ML_mdl.evaluate_model(mdl, viz=False, save=True)

    # Make prediction
    predicted_val = ML_mdl.predict(df)

    # Print results
    acc = abs(100 * (ap - predicted_val) / ap)
    print(f"\nReal value: € {ap}. Margin: {acc:.2f} %")


if __name__ == '__main__':
    lookup_worth(verbose=True, debug=True)

