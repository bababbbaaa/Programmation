import time
import numpy as np
import xgboost as xgb
from binance import Client
from binance.exceptions import BinanceAPIException
import requests
from sklearn.model_selection import train_test_split

class AutomatedTrader:
    def __init__(self, api_key, secret_key):
        self.client = Client(api_key, secret_key)
        self.timeframe = Client.KLINE_INTERVAL_1MINUTE
        self.api_key = api_key
        self.secret_key = secret_key
        self.profit = 0
        self.starting_balance = float(self.client.get_asset_balance(asset='USDT')['free'])  # Valeur de départ de votre balance en USDT
        self.chosen_symbol = None # define chosen_symbol as an instance variable

    def print_current_balance(self):
        try:
            balance = float(self.client.get_asset_balance(asset='USDT')['free'])
            print(f"Current balance: {balance} USDT")
        except BinanceAPIException as e:
            print(f"An error occurred while fetching the current balance: {e}")


    def get_symbols(self):
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
        exchange_info = response.json()

        response = requests.get("https://api.binance.com/api/v1/ticker/24hr")
        symbols = [symbol['symbol'] for symbol in sorted(response.json(), key=lambda x: float(x['volume']), reverse=True)[:10]]
        usdt_symbols = [symbol for symbol in symbols if symbol.endswith('USDT')]

        return usdt_symbols
    
    def get_historical_data(self, symbol):
        try:
            ohlcv = self.client.get_klines(symbol=symbol, interval=self.timeframe)
            close_prices = [float(candle[4]) for candle in ohlcv]
            return close_prices
        except BinanceAPIException as e:
            print(f"An error occurred while fetching historical data for {symbol}: {e}")
            return None
        
    def choose_symbol(self):
        symbols = self.get_symbols()
        print("Available symbols:")
        for i, symbol in enumerate(symbols, 1):
            print(f"{i}. {symbol}")

        while True:
            try:
                choice = int(input("Choose the symbol you want to work with (enter the corresponding number): "))
                if 1 <= choice <= len(symbols):
                    return symbols[choice - 1]
                else:
                    print("Invalid choice. Please enter a number between 1 and the number of available symbols.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")



    def predict_price(self, symbol):
        data = self.get_historical_data(symbol)
        n = len(data)
        X = np.array(range(n)).reshape(-1, 1)
        y = np.array(data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=5,
            reg_alpha=0.5,
            reg_lambda=0.5,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

        next_time = n
        return model.predict(np.array([[next_time]]))[0]



    def get_min_qty(self, symbol_info):
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                return float(filter['minQty'])
        return None

    def get_min_notional(self, symbol_info):
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'MIN_NOTIONAL':
                return float(filter['minNotional'])
        return None

    def get_max_precision(self, symbol_info):
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                return int(-1 * np.log10(float(filter['stepSize'])))
        return None

    def get_current_prices(self):
        symbols = self.get_symbols()
        current_prices = {}
        for symbol in symbols:
            try:
                price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
                current_prices[symbol] = price
            except BinanceAPIException as e:
                print(f"An error occurred while fetching the current price for {symbol}: {e}")
                current_prices[symbol] = None
        return current_prices.get(self.chosen_symbol, {})



    def place_order(self, symbol, side, usdt_quantity=None, stop_loss=None, take_profit=None):
        if usdt_quantity is None:
            return None
        usdt_quantity = float(usdt_quantity)        
        symbol_info = self.client.get_symbol_info(symbol)
        min_value = None
        if symbol_info is None or symbol_info['status'] != 'TRADING':
            print(f"{symbol} is not available for trading")
            return

        min_qty = self.get_min_qty(symbol_info)
        min_notional = self.get_min_notional(symbol_info)
        max_precision = self.get_max_precision(symbol_info)

        current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        
        if min_value is None:
            min_value = min_notional
            
        # Calculate the quantity required to reach the minimum value (plus 1 USDT)
        quantity = (min_value + 1) / current_price
        
        # Adjust the quantity to match the maximum precision allowed
        quantity = round(quantity, max_precision)
        
        if quantity < min_qty:
            print(f"Quantity is below the minimum for trading of {min_qty} {symbol}")
            return

        # Check if the balance is sufficient
        balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        usdt_quantity = current_price * quantity
        usdt_quantity = float(usdt_quantity)

        if usdt_quantity > balance:
            print(f"Insufficient balance for requested action.")
            return

        try:
            if side == "buy":
                order = self.client.order_market_buy(symbol=symbol, quoteOrderQty=round(float(usdt_quantity), 8))

                if stop_loss and take_profit:
                    stop_loss_order = self.client.create_oco_order(
                        symbol=symbol,
                        side="sell",
                        stopLimitPrice=round(current_price * (1 - stop_loss), 8),
                        stopPrice=round(current_price * (1 - stop_loss) * 0.99, 8),
                        quantity=quantity,
                        price=round(current_price * (1 + take_profit), 8),
                    )
            elif side == "sell":
                order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
        except BinanceAPIException as e:
            print(f"An error occurred while placing the order: {e}")
            return

        return order
    

    def should_place_buy_order(self):
        """
        Determines whether a new buy order should be placed based on current market conditions.
        """
        current_prices = self.get_current_prices()

        if None in current_prices.values():
            print("Error fetching current prices for some symbols, skipping trade...")
            return False

        # Get the predicted prices for all symbols
        predicted_prices = {}
        for symbol in current_prices:
            predicted_price = self.predict_price(symbol)
            predicted_prices[symbol] = predicted_price

        # Determine the symbol with the highest predicted price
        best_symbol = max(predicted_prices, key=predicted_prices.get)

        # Check if the predicted price for the best symbol is higher than the current price
        if predicted_prices[best_symbol] > current_prices[best_symbol]:
            return True
        else:
            return False


    def calculate_profit(self, start_time):
        balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        profit = balance - self.starting_balance
        profit_percent = profit / self.starting_balance * 100
        return profit_percent




    def trade(self):
        # Get the symbol you want to work with
        self.chosen_symbol = self.choose_symbol()

        # Get the minimum notional value for the chosen symbol
        chosen_symbol_info = self.client.get_symbol_info(self.chosen_symbol)
        min_notional = self.get_min_notional(chosen_symbol_info)

        # Check if we have enough starting balance to trade
        if self.starting_balance < min_notional:
            print("Starting balance is below the minimum notional value for trading. Exiting...")
            return

        while True:
            try:
                # get current market prices
                current_price = self.get_current_prices()
                
                # check if we have predicted prices for all symbols
                if not isinstance(current_price, dict):
                    print("Error fetching current prices, skipping trade...")
                    continue

                # check if the chosen symbol is present in the current prices
                if self.chosen_symbol not in current_price:
                    print(f"{self.chosen_symbol} price not available, skipping trade...")
                    continue

                # check if it's time to place a new buy order
                if self.should_place_buy_order():
                    # calculate the amount of USDT to spend based on the current balance and risk_factor
                    usdt_quantity = (self.balance * self.risk_factor) / current_price[self.chosen_symbol]
                    min_value = self.get_min_value(self.chosen_symbol)

                    # ensure the usdt_quantity is at least min_value + 1 USDT
                    if min_value is None:
                        min_value = 0
                    usdt_quantity = max(min_value + 1, usdt_quantity)

                    # place buy order and update balance
                    order = self.place_order(self.chosen_symbol, "buy", usdt_quantity=usdt_quantity)
                    if order is not None:
                        self.balance -= order['cummulativeQuoteQty']

                # check if it's time to place a new sell order
                elif self.should_place_sell_order():
                    # calculate the amount of crypto to sell based on the current balance and risk_factor
                    quantity = (self.balance * self.risk_factor) / current_price[self.chosen_symbol]
                    min_quantity = self.get_min_quantity(self.chosen_symbol)

                    # ensure the quantity is at least min_quantity
                    if min_quantity is None:
                        min_quantity = 0
                    quantity = max(min_quantity, quantity)

                    # place sell order and update balance
                    order = self.place_order(self.chosen_symbol, "sell", quantity=quantity)
                    if order is not None:
                        self.balance += order['cummulativeQuoteQty']

                # check if we should sell due to loss
                elif self.should_sell_loss():
                    for symbol in self.bought_symbols:
                        price = self.get_current_price(symbol)
                        if price is not None and self.has_loss(symbol, price):
                            quantity = self.get_owned_quantity(symbol)
                            order = self.place_order(symbol, "sell", quantity=quantity)
                            if order is not None:
                                self.balance += order['cummulativeQuoteQty']
            except BinanceAPIException as e:
                print(f"An error occurred while trading: {e}")
                time.sleep(5)  # wait a certain time before retrying
            except KeyboardInterrupt:
                print("Trading interrupted by user.")
                break

            # calculate and print current profit
            profit_percent = self.calculate_profit(time.time() - self.start_time)
            print(f"Current profit: {profit_percent:.2f}%")

            # wait for next iteration
            time.sleep(self.trade_interval)






