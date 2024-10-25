from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
from flask_cors import CORS
from datetime import datetime, timedelta
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Function to fetch stock data
def get_stock_data(symbol, start='2020-01-01', end=None):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    logging.info(f"Fetching stock data for symbol: {symbol} from {start} to {end}")
    stock_data = yf.download(symbol, start=start, end=end)
    
    if stock_data.empty:
        raise ValueError("No data found for the provided stock symbol.")
    
    stock_data = stock_data[['Close']]  # Only keep the 'Close' column
    stock_data = stock_data.ffill()  # Fill missing data (forward fill)
    logging.info(f"Stock data fetched successfully. Data size: {len(stock_data)} rows")
    return stock_data

# Function to train the model and predict Buy/Sell/Hold
def train_model(stock_data, model_type='linear'):
    logging.info(f"Training model using {model_type} model")
    
    # Create the 'Prediction' column (next day's close price)
    stock_data['Prediction'] = stock_data[['Close']].shift(-1)
    
    # Define features (X) and labels (y)
    X = stock_data.drop(['Prediction'], axis=1).values[:-1]  # Closing price as features
    y = stock_data['Prediction'].values[:-1]  # Next day's closing price as labels

    # If there isn't enough data to make a prediction
    if len(X) < 1:
        raise ValueError("Not enough data to train the model.")
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Select the model type (Linear Regression or Random Forest)
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100)
    
    # Train the model
    model.fit(X_train, y_train)

    # Now we predict the next day's stock price using the last available close price (1 feature)
    last_close_price = stock_data['Close'].values[-1].reshape(-1, 1)  # Reshape for single feature
    future_price = model.predict(last_close_price)[0]  # Get future price as a single value
    current_price = last_close_price[0][0]  # Current closing price

    logging.info(f"Prediction completed. Future price: {future_price}")

    # Calculate stop loss percentage as 2% of the current price
    stop_loss_percentage = 0.02 * current_price
    stop_loss_price = current_price - stop_loss_percentage

    # Determine recommendation based on price change
    price_diff = future_price - current_price
    threshold = 0.01 * current_price  # 1% threshold to avoid small fluctuations
    if price_diff > threshold:
        recommendation = "Buy"
    elif price_diff < -threshold:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    return recommendation, future_price, current_price, stop_loss_price

# Route for the homepage
@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'home.html')

# API route to predict stock prices with Buy/Sell/Hold recommendation
@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Received a request for prediction")
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        model_type = data.get('model_type', 'linear').lower()

        if not symbol:
            logging.warning("No stock symbol provided")
            return jsonify({'error': 'Please provide a valid stock symbol.'}), 400

        # Fetch stock data for the last 30 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        stock_data = get_stock_data(symbol, start=start_date, end=end_date)

        # Get prediction and stop loss
        recommendation, future_price, current_price, stop_loss_price = train_model(stock_data, model_type=model_type)
        logging.info(f"Recommendation for {symbol}: {recommendation}")

        # Determine currency based on symbol
        currency = '₹' if symbol.endswith('.NS') else '$'  # Indian Rupee for NSE stocks

        response = jsonify({
            'prediction': f"The recommendation for {symbol} is to '{recommendation}'.",
            'details': f"Predicted future price: {currency}{future_price:.2f}, Current price: {currency}{current_price:.2f}",
            'stop_loss': f"Suggested Stop Loss Price: {currency}{stop_loss_price:.2f}"
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')

        return response

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5500))
    app.run(host='0.0.0.0', port=port, debug=True)
