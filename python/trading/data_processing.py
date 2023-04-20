import pandas as pd
import numpy as np

def preprocess_data(data):
    """
    Fonction de prétraitement des données historiques pour les adapter aux modèles de prédiction.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données historiques des prix.
        
    Returns:
        pd.DataFrame: DataFrame avec les données prétraitées.
    """
    # Convertir les dates en format datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
    # Trier les données par date
    data = data.sort_values('timestamp')
    
    # Réinitialiser l'index
    data.reset_index(drop=True, inplace=True)
    
    # Retourner les données prétraitées
    return data

def feature_engineering(data, window_sizes=[5, 10, 20, 50]):
    """
    Fonction pour l'ingénierie des caractéristiques à partir des données historiques.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données historiques des prix.
        window_sizes (list): Liste des tailles de fenêtre pour les moyennes mobiles.
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles caractéristiques.
    """
    # Calculer les moyennes mobiles pour différentes tailles de fenêtre
    for window_size in window_sizes:
        data[f'sma_{window_size}'] = data['close'].rolling(window=window_size).mean()
        
    # Calculer les rendements quotidiens en pourcentage
    data['daily_return'] = data['close'].pct_change()
    
    # Calculer la volatilité historique (écart-type des rendements) pour différentes tailles de fenêtre
    for window_size in window_sizes:
        data[f'volatility_{window_size}'] = data['daily_return'].rolling(window=window_size).std()
    
    # Supprimer les lignes avec des valeurs manquantes
    data = data.dropna()
    
    # Réinitialiser l'index
    data.reset_index(drop=True, inplace=True)
    
    # Retourner les données avec les nouvelles caractéristiques
    return data
