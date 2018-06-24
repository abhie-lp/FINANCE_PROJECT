# STOCK MARKET DATA
# The data will be fetched from https://markets.financialcontent.com only
# Initially the data fetched is of Google, Microsoft and Apple

import pandas as pd
import bs4 as bs
import urllib.request as ur
import matplotlib.pyplot as plt
from matplotlib import style

# Function to get the page source of the desired company
def get_page_source(url):
    with ur.urlopen(url) as page_source:
        return page_source.read()

# Scraping the data for the desired company
def scraping(company):
    scrape_data = []											# To store the values of the stock data
    # default page-url that will be used to get the data of different companies
    page_url = "https://markets.financialcontent.com/stocks/quote/historical?Symbol=%s&Month=6&Year=2018&Range=12"
    source = get_page_source(page_url % company)				# To store the page-source of the companies data
    soup = bs.BeautifulSoup(source, "lxml")						# converiting the page-source to BeautifulSoup object to start scraping
    table_data = soup.find("table", class_="quote_detailed_price_table data").find_all("tr")	# all table data of that class and tr tag
    for row in table_data:
        r_data = row.find_all("td")								# Storing all td tag
        scrape_data.append([data.text for data in r_data])
    return scrape_data

# Creating the data frame from the scraped data to do further analysis
def create_dframe(scrape_data):
    return pd.DataFrame(scrape_data, columns=["Date", "Open", "High", "Low", "Close", "Volume", "Change(%)"]).set_index("Date")

# Store the data frame locally for  future use
def to_csv(frame, name):
    frame.to_csv(f"{name}_data.csv", encoding="utf-8")


companies = ("AAPL", "GOOGL", "MSFT")							# Name of the companies whose data will be scraped and analysed


for company in companies:
    scrape_data = scraping(company)								# Function call to start the scraping of the page
    frame = create_dframe(scrape_data[1:])						# Creating DataFrame from index 1 as 0 is blank
    to_csv(frame, company)										# Storing the data as .csv to analyse it in future