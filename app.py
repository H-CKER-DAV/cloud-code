from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

INDIAN_STOCKS = { ... }  # (Use the original dictionary provided)

# Function to fetch stock data
enddt = datetime.now().strftime('%Y-%m-%d')
def get_stock_data(symbol, start='2010-01-01', end=enddt):
    stock_data = yf.download(symbol, start=start, end=end)
    if stock_data.empty:
        raise ValueError("No data found for the provided stock symbol.")
    return stock_data[['Close']].ffill()

# Train model and predict
def train_model(stock_data, prediction_days=15):
    stock_data['Prediction'] = stock_data[['Close']].shift(-prediction_days)
    X = stock_data.drop(['Prediction'], axis=1).values[:-prediction_days]
    y = stock_data['Prediction'].values[:-prediction_days]
    if len(X) < prediction_days:
        raise ValueError("Not enough data to train the model.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    last_close_price = stock_data['Close'].values[-1].reshape(-1, 1)
    future_price = model.predict(last_close_price)[0]
    current_price = last_close_price[0][0]

    price_diff = future_price - current_price
    threshold = 0.01 * current_price
    if price_diff > threshold:
        recommendation = "Buy"
    elif price_diff < -threshold:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    stop_loss = current_price - (0.03 * current_price)  # Stop loss at 3% below current price
    return recommendation, future_price, current_price, stop_loss

@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        if not symbol:
            return jsonify({'error': 'Please provide a valid stock symbol.'}), 400
        stock_data = get_stock_data(symbol)
        recommendation, future_price, current_price, stop_loss = train_model(stock_data)
        
        currency = '₹' if symbol.endswith('.NS') else '$'
        return jsonify({
            'prediction': recommendation,
            'future_price': f"{currency}{future_price:.2f}",
            'current_price': f"{currency}{current_price:.2f}",
            'stop_loss': f"{currency}{stop_loss:.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5500)), debug=True)
