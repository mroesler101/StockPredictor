def days(days): #prediction after days are given
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

  print(y_pred)
  print("SVR score:", regressor.score(x_test,y_test))

  plt.title('Support Vector Regression Model: Days V. Predicted Adj. Closing Price')
  plt.ylabel('Predicted Stock Price Average ($)')
  plt.xlabel('Time (Days)')
  plt.plot(y_pred)
  count=y_pred[0]
  day=0
  max = y_pred[0]
  for i in range(0,days-1):
    if (y_pred[i] < y_pred[i+1]):
      max = y_pred[i+1]
      day = i+2 # for the accurate number of days and not the array number
  print("The highest predicted price in", days, "days is day", day, "with a predicted price of", max)
  if (day==0):
    print("This means that the stock price is predicted to go down in the next", days, "days")
