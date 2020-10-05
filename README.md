![Ironhack logo](https://i.imgur.com/1QgrNNw.png)

# House price prediction

A tool to predict the asking price of Dutch real estate
<br>Ironhack Berlin final project
<br>Rinze Douma
<br>18/09/2020

## Content
- [Project description](#Project-description)
- [Workflow](#Workflow)
- [Organization](#Organization)
- [Links](#Links)

## Project description
I wanted to create a tool to check what a real estate listing in Amsterdam should cost. I've scraped a lot of data from the Dutch housing website www.funda.nl and trained a model to predict the price based on certain characteristics. 

## Workflow
***

## Issues
- make the scraper work
- import module
- determining what factors to use


## Assumptions
data ignored:
- geen uitpond huurwoningen
- no certificates considered



## Organization

#### Data
All raw, intermediate and final data is in the directory 'data'. Scraped data is in .json format, geographical information as shape files in the underlying 'geo' directory and output from dataframes are stored as pickles (.pkl).

#### Notebooks
There are 3 Jupyter notebooks:
* data_cleaning.ipynb - the manual process of cleaning the scraped data
* json-to-df.ipynb - calling of functions that fully manage the cleaning of data and combining current and sold properties
* explore_data.ipynb - a notebook used to explore the data and to assess what preprocessing needed to be done

Also, there are 3 Python scripts:
* json_dataframe.py - the main script to open, clean and export the scraped data
* helpers.py - helper functions for process above
* modelling.py - defines a class that preprocesses the data 

#### Other material


***

## Links
- [Repository](https://github.com/therinz/unemployment_stats)
- [Dataset](https://ec.europa.eu/eurostat/web/lfs/data/database)
- [Trello board](https://trello.com/b/hwqlwb8C)
- [Funda houses](https://www.funda.nl)
- [Defintion of house types (in Dutch](https://www.vhmmakelaars.nl/woning-definities)
