import scrapy


class FundaSpider(scrapy.Spider):
    name = "funda"
    start_urls = ["https://www.funda.nl/koop/amsterdam"]

    def parse(self, response):

        # Find links to houses
        houses = (response
                  .css("div.search-result__header-title-col a::attr(href)")
                  .getall())
        yield from response.follow_all(houses, self.parse_house)

        # Find next page and rerun this method
        next_page = response.selector.xpath("//a[@rel='next']/@href").get()
        yield from response.follow_all(next_page, self.parse)

    def parse_house(self, response):
        response.xpath("//dl[contains(@class, 'object-kenmerken-list')]").getall()
        table = response.xpath("//div[contains(@class, 'object-kenmerken-body')]")

        response.xpath("//dt").getall()
        response.xpath("//dt//following::dd/text()").get()

        table.xpath("//following::dd[1]/text()").get()
        pass

        dt = table.xpath("//dt/text()").getall()
        both = dict(zip(dt, dd))
        dt = [x.strip(" \r\n") for x in dt if x.strip(" \r\n")]

        for dt in table.xpath("//dt/").getall():
            dt.xpath("/following-sibling::dd[0]/text()").get()
