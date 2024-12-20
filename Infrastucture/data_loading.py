import pandas as pd
import os

def load_data(chemin_dossier):
    """
    Charge les données depuis plusieurs fichiers CSV dans un dossier.
    Arguments:
    - chemin_dossier : str : Chemin vers le dossier contenant les fichiers CSV.
    Retourne:
    - DataFrame pandas combinant les données de tous les fichiers.
    """
    fichiers = [f for f in os.listdir(chemin_dossier) if f.endswith('.csv')]
    dfs = []
    for fichier in fichiers:  #Pour chaque saison :
        chemin_fichier = os.path.join(chemin_dossier, fichier) #Liste des noms de l'ensemble des saisons
        df = pd.read_csv(chemin_fichier, delimiter=';')  # Gestion du séparateur 
        df['Année'] = int(fichier.split('.')[0])  # Ajouter une colonne Année à partir du nom du fichier
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) #Concaténation en un seul dataframe
  