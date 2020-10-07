# (c) 2020 Rinze Douma

import os
import scrapy

import pandas as pd
from scraping.scraping.spiders.funda_spider import value_block
from scrapy.crawler import CrawlerProcess

from notebooks.helpers import validate_input
from notebooks.modelling import MachineLearnModel
from notebooks.json_dataframe import APARTMENTS, clean_dataset


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

    if debug:
        df = pd.read_pickle(os.path.join("data", "predict.pkl"))
    else:
        # Cleanup before we start
        if os.path.exists(JSON):
            os.remove(JSON)

        # Ask for type of lookup
        prompt = "Whats the URL of the house? "
        url = validate_input(prompt, type_=str, min_=10).strip(" \"\'")

        # Get data from funda
        settings = {"FEEDS": {JSON: {"format": "json"}},
                    "LOG_ENABLED": True}
        if verbose:
            print("Retrieving data from funda.nl...")
        else:
            settings.update({"LOG_LEVEL": "WARNING"})

        process = CrawlerProcess(settings)
        process.crawl(PredictSpider, link=url)
        process.start()

        # Clean the data
        df = (clean_dataset(JSON, predict=True, verbose=verbose)
              .drop(columns=["asking_price"]))
        df.to_pickle(os.path.join("data", "predict.pkl"))

    apartments = [col for col in df.columns
                  if col.startswith("pt") and col in APARTMENTS]
    try:
        mode = df[apartments].apply(any, axis=1)[0]
    except KeyError:
        mode = False
    print(mode)

    # Train model
    ML_mdl = MachineLearnModel(apartment=mode, verbose=verbose)
    ML_mdl.evaluate_model("EN", viz=False, save=True)

    # Make prediction
    ML_mdl.predict(df)

    # Cleanup
    os.remove(JSON)


if __name__ == '__main__':
    lookup_worth(verbose=True, debug=True)
