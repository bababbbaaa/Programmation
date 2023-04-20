import numpy as np

def calculate_position_size(account_balance, risk_percentage, stop_loss_distance):
    """
    Calcule la taille de la position en fonction de l'équilibre du compte, du pourcentage de risque et de la distance de stop-loss.

    Args:
        account_balance (float): Solde du compte.
        risk_percentage (float): Pourcentage de risque par transaction (0.01 pour 1%).
        stop_loss_distance (float): Distance de stop-loss en termes de prix.

    Returns:
        float: Taille de la position.
    """
    risk_amount = account_balance * risk_percentage
    position_size = risk_amount / stop_loss_distance

    return position_size

def calculate_stop_loss_price(entry_price, position_size, risk_direction):
    """
    Calcule le prix du stop-loss en fonction du prix d'entrée, de la taille de la position et de la direction du risque (long ou short).

    Args:
        entry_price (float): Prix d'entrée de la position.
        position_size (float): Taille de la position.
        risk_direction (str): Direction du risque, 'long' ou 'short'.

    Returns:
        float: Prix du stop-loss.
    """
    if risk_direction == 'long':
        stop_loss_price = entry_price - (position_size / entry_price)
    elif risk_direction == 'short':
        stop_loss_price = entry_price + (position_size / entry_price)
    else:
        raise ValueError("risk_direction doit être 'long' ou 'short'")

    return stop_loss_price

def calculate_take_profit_price(entry_price, risk_reward_ratio, risk_direction):
    """
    Calcule le prix du take-profit en fonction du prix d'entrée, du ratio risque-récompense et de la direction du risque (long ou short).

    Args:
        entry_price (float): Prix d'entrée de la position.
        risk_reward_ratio (float): Ratio risque-récompense.
        risk_direction (str): Direction du risque, 'long' ou 'short'.

    Returns:
        float: Prix du take-profit.
    """
    if risk_direction == 'long':
        take_profit_price = entry_price * (1 + risk_reward_ratio)
    elif risk_direction == 'short':
        take_profit_price = entry_price * (1 - risk_reward_ratio)
    else:
        raise ValueError("risk_direction doit être 'long' ou 'short'")

    return take_profit_price

if __name__ == "__main__":
    account_balance = 10000
    risk_percentage = 0.01
    entry_price = 100
    stop_loss_distance = 5
    risk_direction = 'long'
    risk_reward_ratio = 2

    position_size = calculate_position_size(account_balance, risk_percentage, stop_loss_distance)
    stop_loss_price = calculate_stop_loss_price(entry_price, position_size, risk_direction)
    take_profit_price = calculate_take_profit_price(entry_price, risk_reward_ratio, risk_direction)

    print("Taille de la position :", position_size)
    print("Prix du stop-loss :", stop_loss_price)
    print("Prix du take-profit :", take_profit_price)
