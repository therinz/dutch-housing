import scrapy


class FundaSpider(scrapy.Spider):
    name = "funda"
    start_urls = ["https://www.funda.nl/koop/amsterdam/"]
    allowed_domains = ["funda.nl"]
    custom_settings = {'DEPTH_LIMIT': 5}

    def __init__(self, state="", city="Amsterdam", *args, **kwargs):
        super(FundaSpider, self).__init__(*args, **kwargs)
        if state and state != "verkocht":
            raise CloseSpider("Entered state not valid")
        self.start_urls = [f"http://www.funda.nl/koop/{city}/{state}" ]

    def parse(self, response):

        # Find links to houses
        query = "div.search-result__header-title-col a::attr(href)"
        # Query returns links twice
        houses = [link
                  for i, link in enumerate(response.css(query).getall())
                  if i % 2]
        yield from response.follow_all(houses, self.parse_house)

        # Find next page and rerun this method
        next_page = response.selector.xpath("//a[@rel='next']/@href").get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)

    def parse_house(self, response):
        """Parse information found on house description page."""

        # Retrieve basic info about listing
        subtitle = response.css(".object-header__subtitle::text").get().split()
        info = {"address": response.css(".object-header__title::text").get(),
                "postcode": " ".join(subtitle[0:2]),
                "city": " ".join(subtitle[2:])}

        # Build list of information blocks
        dls = response.xpath("//dl[contains(@class, 'object-kenmerken-list')]")
        info.update({k:v for dl in dls for k, v in value_block(dl).items()})

        yield info

def value_block(dl):
    """Return dictionary with field names & values."""

    # Names of values (left column)
    dts = dl.xpath("./dt")

    # Sold property block
    if (dts[0].xpath("string(./text())").get().strip(" \r\n")
            == "Aangeboden sinds"):
        header = "Verkoopgeschiedenis"
    # Listing characteristics block
    else:
        header = dl.xpath("./preceding-sibling::h3/text()")[-1].get()

    # Dictionary with name: value preceded by header abbreviation
    q_dd = "./following-sibling::dd/text()"
    dd = {header[:3].upper() + "-" +
          dt.xpath("string(./text())").get().strip(" \r\n"):
              dt.xpath(q_dd)[0].get().strip(" \r\n")
          for dt in dts}

    # Extract energy label if current section is Energy
    first_item = next(iter(dd))
    if header == "Energie" and not dd[first_item]:
        dd[first_item] = dl.xpath("./dd/span/text()").get()[0]

    return dd

