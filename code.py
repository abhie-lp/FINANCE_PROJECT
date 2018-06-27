# STOCK MARKET DATA
# The data will be fetched from https://markets.financialcontent.com only
# Initially the data fetched is of Google, Microsoft and Apple

import numpy as np
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
    scrape_data = []									# To store the values of the stock data
    # default page-url that will be used to get the data of different companies
    page_url = "https://markets.financialcontent.com/stocks/quote/historical?Symbol=%s&Month=6&Year=2018&Range=12"
    source = get_page_source(page_url % company)		# To store the page-source of the companies data
    soup = bs.BeautifulSoup(source, "lxml")				# converiting the page-source to BeautifulSoup object to start scraping
    table_data = soup.find("table", class_="quote_detailed_price_table data").find_all("tr")	# all table data of that class and tr tag
    for row in table_data:
        r_data = row.find_all("td")						# Storing all td tag
        scrape_data.append([data.text for data in r_data])
    return scrape_data

# Creating the data frame from the scraped data to do further analysis
def create_dframe(scrape_data):
    return pd.DataFrame(scrape_data, columns=["Date", "Open", "High", "Low", "Close", "Volume", "Change(%)"]).set_index("Date")

# Store the data frame locally for  future use
def to_csv(frame, name):
    frame.to_csv(f"{name}_data.csv", encoding="utf-8")

# Remove the commas from Volume and change its data type to float and remove all NaN rows
def data_cleaning(company):
    company["Volume"] = company["Volume"].str.replace(",", "").astype(float)
    company.dropna(how="any", inplace=True)


company_code = ("AAPL", "GOOGL", "MSFT")					# Name of the companies whose data will be scraped and analysed


for company in company_code:
    scrape_data = scraping(company)						# Get the page-source and scrape the data
    frame = create_dframe(scrape_data[1:])				# Creating DataFrame from index 1 as 0 is blank
    to_csv(frame, company)								# Storing the data as .csv to analyse it in future

# Get the data from local to do analysis and converting index to DateTime format
apple = pd.read_csv("AAPL_data.csv", index_col="Date", parse_dates=True)
google = pd.read_csv("GOOGL_data.csv", index_col="Date", parse_dates=True)
microsoft = pd.read_csv("MSFT_data.csv", index_col="Date", parse_dates=True)

companies = (apple, google, microsoft)

# Converting Volume to float and remove all NaN rows
for i in range(len(companies)):
    data_cleaning(companies[i])
    to_csv(companies[i], company_code[i])               # Storing the changes on local

open_price = pd.DataFrame({"AAPL": apple["Open"],       # DataFrame to store the opening price of all companies
                          "GOOGL": google["Open"],
                          "MSFT": microsoft["Open"]})

open_price.dropna(how="any", inplace=True)              # Removing all the rows w/ NaN values

open_price.plot(secondary_y=["AAPL", "MSFT"], grid=True,)      # Line graph of open_price
plt.show()

close_open = pd.DataFrame({"AAPL": apple["Close"] - apple["Open"],              # DataFrame to store the difference b/w Open and Close
                          "GOOGL": google["Close"] - google["Open"],
                          "MSFT": microsoft["Close"] - microsoft["Open"]})

close_open.dropna(how="any", inplace=True)
close_open.plot(grid=True)
plt.show()

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

companies = ('AAPL_data.csv','GOOGL_data.csv','MSFT_data.csv')
company_name = ('Apple','Google','Microsoft')

for i in range(3):
    df = pd.read_csv(companies[i])
    prices = df['Open'].tolist()
    dates = list(reversed(range(len(prices))))

    # Convert to 1d Vector
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))
    regressor = LinearRegression()
    regressor.fit(dates, prices)
    
    # Visualize Results
    print('\n\033[1m',("Prediction for "+company_name[i]).center(100))
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.scatter(dates, prices, color='yellow', label= 'Actual Price')    # Plotting initial Data Points
    plt.plot(dates, regressor.predict(dates), color='red', linewidth=2, label = 'Predicted Price')    # Plotting line of Linear Regression
    plt.title('Linear Regression | Time vs. Price')
    plt.legend()
    plt.xlabel('Date Integer')
    plt.show()
 
    # Predict Price on Given Date
    date = len(dates)+1
    predicted_price =regressor.predict(date)
    print("Predicted price :", predicted_price[0][0])
    print("Regression Coefficient :", regressor.coef_[0][0])
    print("Regression Intercept :", regressor.intercept_[0])
