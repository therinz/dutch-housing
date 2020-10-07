# (c) 2020 Rinze Douma

from scraping.scraping.spiders.funda_spider import FundaSpider
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from notebooks.helpers import validate_input

print(type(get_project_settings))

"""def lookup_worth():
    # Ask for type of lookup
    prompt = "Type of listing: apartment or house? "
    options = ["apartment", "a", "house", "h"]
    question = validate_input(prompt, type_=str, options=options)

    # Train model
    mode = question in options[:2]
    ML_mdl = MachineLearnModel("combination.pkl", apartment=mode)
    ML_mdl.evaluate_model("EN", viz=False, save=True, verbose=False)

    # Make prediction
    ML_mdl.predict("predict.json")"""