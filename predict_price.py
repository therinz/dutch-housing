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
        print("part 2")
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


url = "https://www.funda.nl/koop/amsterdam/huis-87491130-zuideinde-355-a/"




def lookup_worth():
    # Ask for type of lookup
    prompt = "Whats the URL of the house? "
    url = validate_input(prompt, type_=str, min_=10)

    # Get data from funda
    process = CrawlerProcess(settings={
        "FEEDS": {
            os.path.join("data", "predict.json"): {"format": "json"}
        }
    })
    process.crawl(PredictSpider, link=url)
    process.start()

    # Clean the data
    df = (clean_dataset(os.path.join(), mode="predict")
          .drop(columns=["asking_price"]))
    apartment = df[APARTMENTS].apply(any, axis=1)


    # Train model
    ML_mdl = MachineLearnModel("combination.pkl", apartment=mode)
    ML_mdl.evaluate_model("EN", viz=False, save=True, verbose=False)

    # Make prediction
    ML_mdl.predict("predict.json")