
import pandas as pd
import bs4 as bs
import urllib.request as ur
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


def get_page_source(url):
    with ur.urlopen(url) as page_source:
        return page_source.read()


def scraping(company):
    scrape_data = []
    page_url = "https://markets.financialcontent.com/stocks/quote/historical?Symbol=%s&Month=6&Year=2018&Range=12"
    source = get_page_source(page_url % company)
    soup = bs.BeautifulSoup(source, "lxml")
    table_data = soup.find("table", class_="quote_detailed_price_table data").find_all("tr")
    for row in table_data:
        r_data = row.find_all("td")
        scrape_data.append([data.text for data in r_data])
    return scrape_data


def create_dframe(scrape_data):
    return pd.DataFrame(scrape_data, columns=["Date", "Open", "High", "Low", "Close", "Volume", "Change(%)"]).set_index("Date")


def data_pickle(frame, name):
    with open(f"{name}_data.pickle", "wb") as pck:
        pickle.dump(frame, pck)


companies = ("AAPL", "GOOGL", "MSFT")


for company in companies:
    scrape_data = scraping(company)
    frame = create_dframe(scrape_data[1:])
    data_pickle(frame, company)

