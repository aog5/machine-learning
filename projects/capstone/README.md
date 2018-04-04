# Predicting Housing Prices in Northeastern Queens, NY
## Udacity Machine Learning Nanodegree
### Capstone Project
### Alison O. Gaby

#### **Links to Data:**
* General Information about NYC Open Data Initiative:
https://opendata.cityofnewyork.us/
* Rolling Sales Data for all Boroughs of New York, NY: http://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page
    * For Queens, NY (clicking link with prompt download of xls file): http://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/rollingsales_queens.xls
   * Glossary of Terms used: http://www1.nyc.gov/assets/finance/downloads/pdf/07pdf/glossary_rsf071607.pdf
* Annualized Sales Update: http://www1.nyc.gov/site/finance/taxes/property-annualized-sales-update.page
    * For Queens, NY (clicking link with prompt download of xls file): http://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/neighborhood_sales/queens_sales_prices.xls

#### **Dataset Info:**
* Rolling Sales Data - This dataset is the City of New York’s Department of Finance’s collection of property listings that sold in the last twelve-month period. Total Records: 27,188 sales
    * This dataset includes the following attributes:
        * BOROUGH
        * NEIGHBORHOOD
        * BUILDING CLASS CATEGORY
        * TAX CLASS AT PRESENT
        * BLOCK
        * LOT
        * EASE-MENT
        * BUILDING CLASS AT PRESENT
        * ADDRESS
        * APARTMENT NUMBER
        * ZIP CODE
        * RESIDENTIAL UNITS
        * COMMERCIAL UNITS
        * TOTAL UNITS
        * LAND SQUARE FEET
        * GROSS SQUARE FEET
        * YEAR BUILT
        * TAX CLASS AT TIME OF SALE
        * BUILDING CLASS AT TIME OF SALE
        * SALE PRICE
        * SALE DATE
        
* Annualized Sales Update- This is a collection of yearly sales information of properties sold in New York City between the years 2005 to 2016. (Only data from the years 2013-2016 was used).
    * This dataset includes the following attributes, grouped by neighborhood, and type of home:
        * NEIGHBORHOOD
        * TYPE OF HOME
        * NUMBER OF SALES
        * LOWEST SALE PRICE
        * AVERAGE SALE PRICE
        * MEDIAN SALE PRICE
        * HIGHEST SALE PRICE
*This dataset was used as benchmark initial analysis.
#### **Software Required:**
Python 3.6 (Anaconda distribution recommended)
