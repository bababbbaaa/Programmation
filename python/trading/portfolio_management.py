import numpy as np
from binance.client import Client

def get_account_balance(client):
    balances = {}
    account_info = client.get_account()
    for asset in account_info['balances']:
        balances[asset['asset']] = float(asset['free'])
    return balances

def allocate_portfolio(client, weights):
    account_balance = get_account_balance(client)
    base_currency = 'USDT'
    total_balance_usdt = account_balance[base_currency]
    allocations = {}
    
    for symbol, weight in weights.items():
        allocations[symbol] = total_balance_usdt * weight
    
    return allocations

if __name__ == "__main__":
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"

    client = Client(api_key, api_secret)

    weights = {'BTC': 0.5, 'ETH': 0.3, 'LTC': 0.2}
    allocations = allocate_portfolio(client, weights)
    print("Allocations initiales:", allocations)
