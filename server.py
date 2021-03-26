from flask import Flask
from flask import jsonify
import connexion
from joblib import load
from nasdaq import nasdaq, timeframe

#load the model

#stock_final = load('ticker2.pkl')

# Create the application instance
#app = connexion.App(__name__, specification_dir="./")

# Read the yaml file to configure the endpoints
#app.add_api("master.yaml")

# create a URL route in our application for "/"

app = Flask(__name__)

@app.route('/timeframe/<start>/<end>')
def update(start=None, end=None):
    if not start or not end:
        return jsonify({"Status" : "Error", "Message" : "Invalid Date Format. Please use the date format as YYYY-MM-DD."})
    start = [int(s) for s in start.split('-')]
    end = [int(s) for s in end.split('-')]
    timeframe(start, end)
    return jsonify({"Status" : "Success", "Message" : "Timeframe Updated"})

@app.route('/nasdaq/<ticker>/<days>')
def get(ticker=None, days=None):
    if not ticker:
        return jsonify({"Status" : "Error", "Message" : "Invalid Format. Please run localhost/nasdaq/<ticker>/<days>/"})
    try:
        pred = nasdaq(ticker, days)
    except FileNotFoundError:
        return jsonify({"Status" : "Error", "Message" : "Please run localhost/update/<start>/<end>"})
    return jsonify({"Status" : "Success", "Ticker" : str(ticker), "Prediction" : str(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
