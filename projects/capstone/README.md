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
* Earned Income Credit by Zip Code obtained from IRS
   * Individual Income Tax ZIP Code Data: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2015-zip-code-data-soi
* Air Quality by Zip Code
   * Neighborhood Air Quality Concentrations: Sulfur Dioxide (SO2) : https://catalog.data.gov/dataset/air-quality-ef520
* Actual data used can be found here: https://public.enigma.com/datasets/new-york-city-property-sales/fd6efa37-2dcd-4294-8795-a0e6044f15b4

#### **Dataset Info:**
* New York City Property Sales - This dataset is the City of New York’s Department of Finance’s collection of property listings that sold in the last twelve-month period. Total Records: 60,295 sales
    * This dataset includes the following attributes:
      * Borough Code
      * Borough Code Definition
      * Neighborhood
      * Building Class Category Code
      * Building Class Category Code Definition
      * Tax Class At Present Code
      * Tax Class At Present Code Definition
      * Building Class At Present Code
      * Building Class At Present Code Definition
      * Address
      * Apartment Number
      * Zipcode
      * Residential Units
      * Commercial Units
      * Total Units
      * Land Square Feet
      * Gross Square Feet
      * Year Built
      * Tax Class At Time Of Sale Code
      * Tax Class At Time Of Sale Code Definition
      * Building Class At Time Of Sale Code
      * Building Class At Time Of Sale Code Definition
      * Sale Price
      * Sale Date
      * Community District
      * Community District Definition
      * School District
      * School District Definition
      * City Council District
      * City Council District Definition
      * Fire Company
      * Fire Company Definition
      * Police Precinct
      * Police Precinct Definition
      * Health Center District
      * Owner Type
      * Owner Type Definition
      * Floor Area - Total
      * Floor Area - Commercial
      * Floor Area - Residential
      * Floor Area - Office
      * Floor Area - Retail
      * Floor Area - Garage
      * Floor Area - Storage
      * Floor Area - Factory
      * Floor Area - Other
      * Number Of Buildings
      * Number Of Floors
      * Number Of Residential Units
      * Number Of Total units
      * Assessed Land Value
      * Assessed Total Value
      * Exempt Land Value
      * Exempt Total Value
      * Year Altered One
      * Year Altered Two
      * Historical District Name
      * Maximum Allowable Residential Floor Area Ratio
      * Maximum Allowable Commercial Floor Area Ratio
      * Maximum Allowable Facility FAR
      * X Coordinate
      * Y Coodrinate
      * Enigma Serial ID

* Annualized Sales Update- This is a collection of yearly sales information of properties sold in New York City between the years 2005 to 2016. (Only data from the years 2013-2016 was used).
    * This dataset includes the following attributes, grouped by neighborhood, and type of home:
        * NEIGHBORHOOD
        * TYPE OF HOME
        * NUMBER OF SALES
        * LOWEST SALE PRICE
        * AVERAGE SALE PRICE
        * MEDIAN SALE PRICE
        * HIGHEST SALE PRICE
        
* Tax Information - Earned Income Credit by Zip was extracted from this dataset, and mapped to the existing housing data
* Neighborhood Air Quality Concentrations: Sulfur Dioxide (SO2) - this data was extracted from the linked dataset, and mapped from each UHF42 section to each zip code, then mapped to the original housing data.

*This dataset was used as benchmark initial analysis.
#### **Software Required:**
Python 3.6 (Anaconda distribution recommended)
