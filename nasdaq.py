# I used this source to help me find a huge NASDAQ csv file containing stock names: https://towardsdatascience.com/downloading-historical-stock-prices-in-python-93f85f059c1f
# Market categories feature in CSV file: Q = NASDAQ Global Select MarketSM; G = NASDAQ Global MarketSM; S = NASDAQ Capital Market
# Round lot size feature: securities to be traded on exchange (typically 100)

import yfinance as yf # Yahoo finance will provide historical stock prices to use for the predictions
import datetime
import time
import requests
import io
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from joblib import dump, load 
from sklearn.preprocessing import StandardScaler
import json

def collectDataframe(start, end):
  # List of stocks in NASDAQ found from towardsdatascience.com to have a collection of stocks to form a model
  url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
  s = requests.get(url).content
  companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
  Symbols = companies['Symbol'].tolist() #create list for the various companies
  start = start.split('-')
  end = end.split('-')
  sty = int(start[0])
  stm = int(start[1])
  std = int(start[2])
  ey = int(end[0])
  em = int(end[1])
  ed = int(end[2])
  if (sty > ey):
    return("Start year must be lower than end year")
  if (sty == ey):
    if (stm > em):
      return("End month has to be after start month")
    if (stm == em):
      if (std > ed):
        return("End day has to be after start day")
  start = (sty, stm, std)
  end = (ey, em, ed)
  start = datetime.datetime(sty,stm,std) # Start and end time to look at stock history (1 month from november-december 2020 in this model)
  end = datetime.datetime(ey,em,ed)
# use arguments to find date 

# This creates an empty list
  stock_final = pd.DataFrame()
# Go through every symbol that we added to the list
  for i in Symbols:  
    
    # This prints each symbol that is being acquired for
    print( str(Symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)  
    
    try:
        # Yahoo Finance - finds and collects the information needed to complete the dataframe and list for predictions
        stock = []
        stock = yf.download(i,start=start, end=end, progress=False)
        
        # This appends the individual stock prices to the list
        if len(stock) == 0:
            None
        else:
            stock['Name']=i
            stock_final = stock_final.append(stock,sort=False)
    except Exception:
        None
  dump(stock_final, 'ticker.pkl')
  return "Update NASDAQ Stock Tickers: Successful"

def timeframe(start, end):
  start = start.split('-')
  end = end.split('-')
  sty = int(start[0])
  stm = int(start[1])
  std = int(start[2])
  ey = int(end[0])
  em = int(end[1])
  ed = int(end[2])
  if (sty > ey):
    return("Start year must be lower than end year")
  if (sty == ey):
    if (stm > em):
      return("End month has to be after start month")
    if (stm == em):
      if (std > ed):
        return("End day has to be after start day")
  start = (sty, stm, std)
  end = (ey, em, ed)
  with open('start.json', 'w') as startF:
    json.dump(start, startF)
  with open('end.json', 'w') as endF:
    json.dump(end, endF)
  times = []
  times.append(f"Time Update of Adjusted Stock Prices: Successful")
  times.append(f"New Start Time: {start}")
  times.append(f"New End Time: {end}")
  return times

def nasdaq(ticker, days):
  stock_final = load('ticker.pkl')
  with open('start.json') as startf:
    start = json.load(startf)
  with open('end.json') as endf:
    end = json.load(endf)
  start = datetime.datetime(start[0], start[1], start[2]) # Start and end time to look at stock history (1 month from november-december 2020 in this model)
  end = datetime.datetime(end[0], end[1], end[2])
  days = int(days)
  if (days < 1):
    return("Invalid number of days to predict")
  if stock_final.query("Name == '{}'".format(ticker)).empty:
    return("Invalid NASDAQ ticker or information not found")
  st = yf.download(ticker,start=start, end=end, progress=False)
  
  #Makes another column called prediction
  st['Prediction'] = st[['Adj Close']].shift(-days)

  X = np.array(st.drop(['Prediction'],1))

#Take the last 'days' rows to predict the next # of days
  X = X[:-days]
#print(X)

  y = np.array(st['Prediction'])
# Get all of the y values except the last 'days' of rows
  y = y[:-days]
#print(y)

#Train 80% and test 20% of the data
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# drops all days before the prediction column, saves only # of days to graph
  stockprice = np.array(st.drop(['Prediction'],1))[-days:]

  regressor = SVR(kernel = 'poly')
  regressor.fit(x_train, y_train)
  y_pred = regressor.predict(stockprice)

  count=y_pred[0]
  day=0
  max = y_pred[0]
  for i in range(0,days):
    if (y_pred[i] > max):
      max = y_pred[i]
      day = i+1 # for the accurate number of days and not the array number
  analysis = []
  analysis.append(f"The highest predicted price of {ticker} in {days} days is day {day} with a predicted price of ${round(max, 2)} and an SVR score of: {regressor.score(x_test,y_test)}")
  if (day==0):
    analysis.append(f"This means that {ticker} is predicted to go down in the next {days} days")
# When day 0 is the lowest, that means that the stock price is predicted to go down
  
  analysis.append(f"Prediction: {str(y_pred)}")
  return analysis
