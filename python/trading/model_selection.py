import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def split_data(data, target_column, test_size=0.2, random_state=None):
    """
    Fonction pour diviser les données en ensembles d'entraînement et de test.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données avec les caractéristiques et la cible.
        target_column (str): Nom de la colonne cible (variable à prédire).
        test_size (float): Proportion de l'ensemble de test (entre 0 et 1).
        random_state (int): Graine pour la génération de nombres aléatoires.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test), les ensembles de données d'entraînement et de test.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Fonction pour entraîner et évaluer un modèle de Machine Learning.
    
    Args:
        model (sklearn.base.BaseEstimator): Modèle de Machine Learning à entraîner et évaluer.
        X_train (pd.DataFrame): Données d'entraînement.
        X_test (pd.DataFrame): Données de test.
        y_train (pd.Series): Cibles d'entraînement.
        y_test (pd.Series): Cibles de test.
        
    Returns:
        dict: Dictionnaire contenant les scores d'évaluation du modèle.
    """
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    evaluation = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae
    }

    return evaluation

def select_best_model(X_train, X_test, y_train, y_test, models, param_grids):
    """
    Fonction pour sélectionner le meilleur modèle en utilisant une recherche sur grille (GridSearchCV).
    
    Args:
        X_train (pd.DataFrame): Données d'entraînement.
        X_test (pd.DataFrame): Données de test.
        y_train (pd.Series): Cibles d'entraînement.
        y_test (pd.Series): Cibles de test.
        models (list): Liste des modèles à évaluer.
        param_grids (list): Liste des grilles de paramètres pour chaque modèle.
        
    Returns:
        dict: Dictionnaire contenant les résultats de chaque modèle et le modèle sélectionné.
    """
    best_model = None
    best_score = float('inf')

    for model, param_grid in zip(models, param_grids):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)

        score = -grid_search.best_score_

        if score < best_score:
            best_score = score
            best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)

    results = train_and_evaluate_model(best_model, X_train, X_test, y_train, y_test)
    results['best_model'] = best_model

    return results

if __name__ == "__main__":
    # Charger les données et les diviser en ensembles d'entraînement et de test
    data = pd.read_csv('path/to/data.csv')
    X_train, X_test, y_train, y_test = split_data(data, target_column='your_target_column')

    # Définir les modèles et les grilles de paramètres pour la sélection de modèles
    models = [
        LinearRegression(),
        RandomForestRegressor(),
        XGBRegressor()
    ]

    param_grids = [
        {},
        {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 7]}
    ]

    # Sélectionner le meilleur modèle
    results = select_best_model(X_train, X_test, y_train, y_test, models, param_grids)

    # Afficher les résultats
    print("Meilleur modèle :", results['best_model'])
    print("RMSE d'entraînement :", results['train_rmse'])
    print("RMSE de test :", results['test_rmse'])
    print("MAE d'entraînement :", results['train_mae'])
    print("MAE de test :", results['test_mae'])
