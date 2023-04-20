import numpy as np
from binance.client import Client
from data_processing import get_historical_data
from model_selection import select_best_model
from risk_management import calculate_risk_metrics
from portfolio_management import allocate_portfolio

def simple_moving_average(client, symbol, interval, window):
    historical_data = get_historical_data(client, symbol, interval)
    sma = historical_data['close'].rolling(window=window).mean()
    return sma

def exponential_moving_average(client, symbol, interval, window):
    historical_data = get_historical_data(client, symbol, interval)
    ema = historical_data['close'].ewm(span=window).mean()
    return ema

def mean_reversion(client, symbol, interval, window):
    historical_data = get_historical_data(client, symbol, interval)
    sma = simple_moving_average(client, symbol, interval, window)
    stddev = historical_data['close'].rolling(window=window).std()
    upper_band = sma + (stddev * 2)
    lower_band = sma - (stddev * 2)
    return upper_band, lower_band

def main():
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"

    client = Client(api_key, api_secret)

    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_1DAY
    window = 50

    best_model = select_best_model(client, symbol, interval)
    risk_metrics = calculate_risk_metrics(client, symbol, interval)
    
    weights = {'BTC': 0.5, 'ETH': 0.3, 'LTC': 0.2}
    allocations = allocate_portfolio(client, weights)
    print("Allocations initiales:", allocations)

    sma = simple_moving_average(client, symbol, interval, window)
    ema = exponential_moving_average(client, symbol, interval, window)
    upper_band, lower_band = mean_reversion(client, symbol, interval, window)

if __name__ == "__main__":
    main()
