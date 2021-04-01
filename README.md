# StockPredictor
This is an adjusted closing stock price predictor for NASDAQ stock tickers within a dataframe. The dataframe endpoint is used to update the NASDAQ stocks into a pickled dataframe from the .csv file with a link in my machine learning model. The timeframe endpoint allows the user to enter a specific start and end time to update the times for the prices extracted from Yahoo Finance into JSON files which are constantly updated from the users input. In the actual predictor endpoint, most stock tickers work for endpoints to download the necessary data through Yahoo Finance's library in python. This is used to create the prediction of a given number of days through Support Vector Regression and provides the highest prediction price with an SVR score to identify the quality of the stock price prediction.
