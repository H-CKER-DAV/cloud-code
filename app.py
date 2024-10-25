from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
from sklearn.linear_model import LinearRegression
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

# Symbol mapping for Indian stocks
INDIAN_STOCKS = {
    "HINDALCO": "HINDALCO.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC": "HDFC.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "LT": "LT.NS",
}

# Function to fetch stock data
def get_stock_data(symbol, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d')):
    stock_data = yf.download(symbol, start=start, end=end)
    if stock_data.empty:
        raise ValueError("No data found for the provided stock symbol.")
    return stock_data[['Close']].ffill()

# Function to train the model and provide Buy/Sell/Hold recommendation
def train_model(stock_data, prediction_days=15, model_type='linear'):
    stock_data['Prediction'] = stock_data[['Close']].shift(-prediction_days)
    X = stock_data.drop(['Prediction'], axis=1).values[:-prediction_days]
    y = stock_data['Prediction'].values[:-prediction_days]

    if len(X) < prediction_days:
        raise ValueError("Not enough data to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression() if model_type == 'linear' else RandomForestRegressor(n_estimators=100)
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

    return recommendation, future_price, current_price

# Function to determine the best period to Buy/Sell within a specified range
def best_period_to_trade(stock_data, start_date, end_date):
    try:
        filtered_data = stock_data.loc[start_date:end_date]
        min_price_row = filtered_data['Close'].idxmin()
        max_price_row = filtered_data['Close'].idxmax()

        if min_price_row < max_price_row:
            return f"Buy on {min_price_row.strftime('%Y-%m-%d')} and Sell on {max_price_row.strftime('%Y-%m-%d')}"
        else:
            return "Hold, as no profitable opportunity was found in this period."
    except Exception as e:
        return f"Error finding trade period: {str(e)}"

# Function to calculate stop-loss price
def calculate_stop_loss(current_price, percent):
    return current_price * (1 - percent / 100)

@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'home.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        model_type = data.get('model_type', 'linear').lower()
        prediction_days = int(data.get('prediction_days', 15))
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        stop_loss_percent = float(data.get('stop_loss_percent', 5))

        if not symbol:
            return jsonify({'error': 'Please provide a valid stock symbol.'}), 400

        symbol = INDIAN_STOCKS.get(symbol, symbol)
        stock_data = get_stock_data(symbol)

        recommendation, future_price, current_price = train_model(stock_data, prediction_days, model_type)
        trade_recommendation = best_period_to_trade(stock_data, start_date, end_date)
        stop_loss_price = calculate_stop_loss(current_price, stop_loss_percent)

        currency = '₹' if symbol.endswith('.NS') else '$'

        return jsonify({
            'prediction': f"The recommendation for {symbol} is to '{recommendation}'.",
            'details': f"Predicted future price: {currency}{future_price:.2f}, Current price: {currency}{current_price:.2f}",
            'trade_recommendation': trade_recommendation,
            'stop_loss': f"Suggested stop-loss price: {currency}{stop_loss_price:.2f}"
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
