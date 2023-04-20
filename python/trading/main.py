from automated_trader import AutomatedTrader

def main():
    # Put your code for initializing API keys and creating an instance of AutomatedTrader here
    api_key = ''
    secret_key = ''
    trader = AutomatedTrader(api_key, secret_key)

    # Get the list of USDT symbols and print them out
    symbols = trader.get_symbols()
    usdt_symbols = [symbol for symbol in symbols if symbol.endswith("USDT")]
    print("USDT symbols available for trading:")
    for i, symbol in enumerate(usdt_symbols):
        print(f"{i + 1}. {symbol}")

    # Ask the user to choose a symbol and start trading
    while True:
        try:
            symbol_index = int(input("Choose the symbol you want to trade (enter the corresponding number): "))
            chosen_symbol = usdt_symbols[symbol_index - 1]
            print(f"Starting automated trading for {chosen_symbol}...")
            trader.trade()
        except (ValueError, IndexError):
            print("Invalid symbol index. Please choose a valid number.")
        except KeyboardInterrupt:
            print("Trading interrupted by user.")
            break

if __name__ == "__main__":
    main()
