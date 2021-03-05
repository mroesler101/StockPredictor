def my_tick(ticker):
    st = stock_final.query("Name == '{}'".format(ticker))
    st = st.drop(columns=['Name'])
    st.plot(y='Adj Close') 
    plt.title('Date V. Actual Adj. Closing Price')
    plt.xlabel('Date (Days)')
    plt.ylabel('Adj. Closing Price ($)')
    plt.show()  
    return st
