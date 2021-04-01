from flask import Flask
from flask import jsonify
import connexion
from joblib import load

#load the model
stock_final = load('ticker.pkl')

# Create the application instance
app = connexion.App(__name__, specification_dir="./")

# Read the yaml file to configure the endpoints
app.add_api("master.yaml")

# create a URL route in our application for "/"
@app.route("/")
def home():
    msg = {"Name": "NASDAQ Adjusted Closing Price Stock Predictor Application by Max Roesler", "Function 1": "Update Timeframe - /timeframe/<start>/<end>", "Function 2": "Get Prediction - /nasdaq/<ticker>/<days>", "Function 3": "Update NASDAQ Tickers (ONLY RUN IF TICKER FILE IS CORRUPT/NEEDS TICKERS) - /dataframe/<start>/<end>"}
    return jsonify(msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
