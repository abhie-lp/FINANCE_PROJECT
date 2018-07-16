# STOCK MARKET DATA
# The data will be fetched from https://markets.financialcontent.com only
# Initially the data fetched is of Google, Microsoft and Apple

import pandas as pd
import numpy as np
import bs4 as bs
import urllib.request as ur
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

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
company_name = ('Apple','Google','Microsoft')

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

close_open.plot(figsize=(15, 10))                       # Graph for close_open
plt.title("Difference in Close and Open", fontsize=20, fontweight="bold")
plt.ylabel("Close - Open ($)")
plt.legend()
plt.show()

close_price = pd.DataFrame({"AAPL": apple["Close"],     # DataFrame to store Closing price of all companies from old to new
                           "GOOGL": google["Close"],
                           "MSFT": microsoft["Close"]})

close_price.dropna(how="any", inplace=True)

# Calculating Relative Returns
rel_returns = close_price.pct_change()

rel_returns.plot(figsize=(15, 10))
plt.ylabel("Percent Change")
plt.title("Relative Returns", fontsize=20, fontweight="bold")
plt.legend(loc="upper left")
plt.show()

# Log Returns
# First taking log of all prices and then there difference
log_returns = np.log(close_price).diff()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
for i in company_code:    
    ax1.plot(log_returns.index, log_returns[i].cumsum(), label=i)
    ax1.set_ylabel("Cumulative log returns")
    ax1.legend(loc="best")
    ax2.plot(log_returns.index, 100*(np.exp(log_returns[i].cumsum()) - 1), label=i)
    ax2.set_ylabel("Total rel. returns")
    ax2.legend(loc="best")

plt.show()

# Calculating Simple Moving Average(SMA)
short_rolling = close_price.rolling(window=20).mean().dropna(how="any")         # SMA for 20 days
long_rolling = close_price.rolling(window=100).mean().dropna(how="any")         # SMA for 100 days

start = str(long_rolling.index[0]).split()[0]           # Date from where plotting to be done

for i in range(3):
    close_price.loc[start:, company_code[i]].plot(x=close_price.loc[start:].index, label="Closing")
    long_rolling.loc[start:, company_code[i]].plot(x=long_rolling.loc[start:].index, label="100-Days SMA")
    short_rolling.loc[start:, company_code[i]].plot(x=short_rolling.loc[start:].index,  label="20-Days SMA", grid=True,figsize=(15, 10))
    plt.ylabel("Price ($)")
    plt.title("SMA for {}".format(company_name[i]), fontsize=20, fontweight="bold")
    plt.legend(loc="upper left")
    plt.show()

# Calculating Exponential Moving Average(EMA)
ema_short = close_price.ewm(span=20, adjust=False).mean()           # EMA for 20 days

start = start = str(short_rolling.index[0]).split()[0]

# Plotting EMA
for i in range(3):
    close_price.loc[start:, company_code[i]].plot(x=close_price.loc[start:].index, label="Closing")
    ema_short.loc[start:, company_code[i]].plot(x=long_rolling.loc[start:].index, label="20-Days EMA")
    short_rolling.loc[start:, company_code[i]].plot(x=short_rolling.loc[start:].index,  label="20-Days SMA", grid=True,figsize=(15, 10))
    plt.ylabel("Price ($)")
    plt.title("{}".format(company_name[i]), fontsize=20, fontweight="bold")
    plt.legend(loc="upper left")
    plt.show()

    

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

companies = ('AAPL_data.csv','GOOGL_data.csv','MSFT_data.csv')

for i in range(3):
    df = pd.read_csv(companies[i])
    prices = df['Open'].tolist()
    dates = list(reversed(range(len(prices))))

    # Convert to 1d Vector
    dates = np.reshape(dates, (len(dates), 1))      # Dates will act as a Feature vector
    prices = np.reshape(prices, (len(prices), 1))   # Prices will act as a Response vector
    
    # Define Linear Regressor Object
    regressor = LinearRegression()
    regressor.fit(dates, prices)    #Fitting a linear model
    
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
