import pandas as pd
from sklearn.impute import KNNImputer

def clean_data(df):
    """
    Nettoie les données en gérant les valeurs manquantes, doublons, colonnes inutiles,
    et en effectuant des transformations spécifiques (ex : remplacement de virgules, imputation KNN).
    Arguments:
    - df : DataFrame pandas : Les données brutes.
    Retourne:
    - DataFrame nettoyé.
    """
    # Supprimer les doublons
    df = df.drop_duplicates()

    # Supprimer les espaces autour des noms de colonnes
    df.columns = df.columns.str.strip()

    # Traitement des colonnes de type 'object' contenant des virgules
    for v in df.select_dtypes(include=['object']).columns:
        if df[v].str.contains(',').any():
            df[v] = df[v].str.replace(',', '').astype(float)

    # Identification des colonnes avec des valeurs nulles
    var_null = []  # Initialisation de la liste des variables contenant des valeurs nulles
    for v in df:  # Pour chaque variable
        verif = df[v].isnull().sum()  # Compter le nombre de valeurs nulles
        if verif > 0:  # Si supérieur à 0
            print(f"{v}: {verif} valeurs nulles")  # Afficher
            var_null.append(v)  # Ajouter à la liste var_null

    # Sélection des variables quantitatives, après avoir retiré les colonnes contenant des valeurs nulles
    var_keep = df.drop(columns=var_null).select_dtypes(include=['float64', 'int64'])

    # Création d'un nouveau DataFrame avec les colonnes ayant des valeurs nulles et les variables quantitatives
    df_knn = pd.concat([df[var_null], var_keep], axis=1)

    # Application de l'imputation KNN (k=5 voisins)
    imputer = KNNImputer(n_neighbors=5)
    df_impute = pd.DataFrame(imputer.fit_transform(df_knn), columns=df_knn.columns)

    # Imputation des nouvelles valeurs dans le DataFrame original
    for v in var_null:
        df[v] = df[v].fillna(df_impute[v])

    # Nettoyage de la colonne 'Team' : suppression des astérisques
    distinct_teams_before = df['Team'].unique()
    print(distinct_teams_before)
    df["Team"] = df["Team"].str.replace('*', '', regex=False)
    distinct_teams_after = df['Team'].unique()
    print(distinct_teams_after)

    # Création des nouvelles variables de performance "W/L" et "PW/PL"
    df["W/L"] = df["W"] / (df["W"] + df["L"])
    df = df.drop(columns=["W", "L"])

    df["PW/PL"] = df["PW"] / (df["PW"] + df["PL"])
    df = df.drop(columns=["PW", "PL"])

    return df



def subsets(df, critere):
    """
    Crée des subsets basés sur un critère spécifique (exemple : rapidité de jeu).
    Arguments:
    - df : DataFrame pandas : Les données complètes.
    - critere : str : Nom de la colonne pour filtrer les subsets.
    Retourne:
    - Une liste de DataFrames correspondant aux subsets.
    """
    mediane = df[critere].median()
    subset_sup = df[df[critere] > mediane]
    subset_inf = df[df[critere] <= mediane]
    return subset_sup, subset_inf