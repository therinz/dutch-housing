![Ironhack logo](https://i.imgur.com/1QgrNNw.png)

# House price prediction

A tool to predict the asking price of houses and apartments in Amsterdam
<br>Ironhack Berlin final project
<br>Rinze Douma
<br>09/10/2020

## Content
- [Project description](#Project-description)
- [Use](#Use)
- [Workflow](#Workflow)
- [Issues encountered](#Issues-encountered)
- [Organization](#Organization)
- [Links](#Links)

## Project description
For this project, I wanted to create a tool to check what a real estate listing in Amsterdam should cost. This way, one could check whether the asking price of a given listing is realistic or not. I've scraped a lot of data from the Dutch housing website www.funda.nl and trained a machine learning model to predict the price based on the characteristics mentioned with each listing. 

## Use
To use this tool, just open `predict_price.py`. The program will ask for the URL of the listing to be questioned. After that, an API key will need to be entered in order to get the coordinates from Google Maps. Finally, the program will return the expected asking price, along with some information to put the numbers into perspective.

An image says a thousand words:
![Screenshot of use of program](https://github.com/therinz/dutch-housing/blob/master/visuals/Screenshot_showcase_Jupyter.png)

## Workflow
1. Initially I started with a feasibility study and what it would take. The various steps were collected on a Trello board.
2. To retrieve the data I had scrape the Funda website, for which I used Scrapy
3. The gathered data was explored in Jupyter notebooks. The functions to clean the dataset were then stored in an external Python script. 
4. Neighborhood shapes were collected from the Amsterdam municipal website. Using the address coordinates were retrieved from Google maps. Coordinates were then binned into neighborhoods with GeoPandas.
4. Initial statistical analysis and pre-processing was performed in Jupyter and then stored centrally in a Python file.
5. Dataset was split into apartments and houses, due to their large differences. 
6. Outliers were removed, split in train and test sets (60/40) and standardized.
7. Various linear regression models were trained to assess feasibility. 
8. One full script created to check a certain Funda URL for expected asking price.

## Issues encountered
- Had to learn how to use Scrapy, GeoPandas, various linear models, residuals plotting and how to work with XML and shape-files.
- Difficult to make scraping work for all cases
- Time management was critical because there are so many characteristics to clean
- For every feature it needed to be decided what extremes and possible values to use
- I also encountered some unexpected issues with Jupyter handling relative imports differently than when directly in a Python script.
- Splitting the dataset caused some weird behavior. This was remedied by resetting the index and copy-pasting, oddly.
- The Yellowbrick module for residuals plotting needs to be updated and throws a future warning.
- The standard of saving files in a pickle has changed. That's why Python version >= 3.8 is needed.
- Running Getpass to input a password breaks the program when run directly in the command line.

## Organization
The root directory contains the main file `predict_price.py`, in which lookup_worth is the main function. Also, there is a Jupyter notebook to demonstrate the working of the program. 

#### Data
All raw, intermediate and final data is in the directory `data`. Scraped data is in .json format, geographical information as shape files in the underlying 'geo' directory and output from dataframes are stored as pickles (.pkl).

#### Notebooks
All notebooks and scripts (expect the main function) are in the directory `notebooks`.

There are 3 Jupyter notebooks:
* `data_cleaning.ipynb` - the manual process of cleaning the scraped data
* `json-to-df.ipynb` - use of functions that fully manage the cleaning of data and combining current and sold properties
* `explore_data.ipynb` - a notebook used to explore the data and to assess what pre-processing needed to be done

Also, here are 3 Python scripts, in which tasks are automated:
* `json_dataframe.py` - the main script to open, clean and export the scraped data
* `helpers.py` - helper functions for process above
* `modelling.py` - defines a class that pre-processes the data and fits a model. The predict method is used to compare a unseen listing with the trained model.

#### Visuals
The folder `visuals` contains all graphical content:
- A Tableau workbook, along with a pdf and png showing the spread of listings used in the dataset
- Graph detailing most correlating features
- Risiduals plots 
- Some visual content for the presentation, among which a screengrab showing the Funda website

#### Scraping
- The `scraping` directory contains all files related to Scrapy. The spider used to scrape the training data is in `scrapy/scraping/spiders/funda_spider.py`. The main function has a variant of this spider that retrieves the data of subject listing.

## Links
- [Repository](https://github.com/therinz/dutch-housing)
- [Trello board](https://trello.com/b/hwqlwb8C)
- [Funda houses](https://www.funda.nl)
- [Definition of house types (in Dutch)](https://www.vhmmakelaars.nl/woning-definities)
