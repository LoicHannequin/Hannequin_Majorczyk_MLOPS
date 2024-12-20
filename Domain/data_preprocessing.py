import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocessing(df, cible):
    """
    Prépare les données pour la modélisation : encode les variables catégoriques et normalise les données.
    Arguments:
    - df : DataFrame pandas : Les données nettoyées.
    - cible : str : Le nom de la colonne cible.
    Retourne:
    - X : Features
    - y : Target
    - scaler : StandardScaler pour la normalisation des nouvelles données.
    """
    y = df[cible]
    X = df.drop(columns=[cible])

    # Encodage des variables catégoriques
    X = pd.get_dummies(X, drop_first=True)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler