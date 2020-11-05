# (c) 2020 Rinze Douma

from pathlib import Path

import scrapy
import pandas as pd
from scraping.scraping.spiders.funda_spider import value_block
from scrapy.crawler import CrawlerProcess

from notebooks.modelling import MachineLearnModel, APARTMENTS
from notebooks.json_dataframe import clean_dataset, validate_input, log_print


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


def lookup_worth(verbose=False, debug=False):
    """Return predicted value of given listing."""

    # Define paths
    base_path = Path(__file__).resolve().parent / "data"
    pickle = base_path / "predict.pkl"
    json = base_path / "predict.json"

    # Debug mode to use stored data
    log_print(f"Debug: {debug}.", verbose)
    if debug:
        df = pd.read_pickle(pickle)
    else:
        # Cleanup before we start
        json.unlink(missing_ok=True)

        # Ask for type of lookup
        prompt = "What's the URL of the house? \n"
        url = ""
        while "funda.nl/" not in url:
            url = validate_input(prompt, type_=str, min_=10).strip(" \"\'")
            prompt = "Not a valid url! Please try again: \n"

        # Set settings for crawler
        rel_path = Path(json.parent.name, json.name)
        settings = {"FEEDS": {rel_path: {"format": "json"}},
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
        df = clean_dataset(json, predict=True, verbose=verbose)

        # Save for future debugging
        df.to_pickle(pickle)

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
            "RI",
            # "LA",
            # "EN"
        ]
        for mdl in mdls:
            ML_mdl.evaluate_model(mdl, viz=False, save=True)
    else:
        mdl = "LA" if mode else "LA"
        ML_mdl.evaluate_model(mdl, viz=False, save=True)

    # Make prediction
    predicted_val = ML_mdl.predict(df)

    # Print results
    acc = abs(100 * (ap - predicted_val) / ap)
    print(f"\nReal value: € {ap}. Margin: {acc:.2f} %")


if __name__ == '__main__':
    lookup_worth(verbose=True, debug=False)
