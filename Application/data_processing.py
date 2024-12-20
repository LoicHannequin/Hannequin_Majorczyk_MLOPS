from Infrastucture.data_loading import load_data
from Infrastucture.data_cleaning import clean_data, subsets
from Domain.data_preprocessing import preprocessing
from Domain.data_modelisation import ModelTrainer
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def sauvegarder_modele(modele, chemin, exper_name, subset):
    """
    Sauvegarde un modèle entraîné dans MLflow.
    Arguments:
    - modele : modèle sklearn : Modèle à sauvegarder.
    - chemin : str : Chemin où sauvegarder le modèle localement.
    - exper_name : str : Nom de l'expérience MLflow.
    - subset : str : Nom du subset utilisé pour l'entraînement.
    """
    mlflow.set_experiment(exper_name)
    with mlflow.start_run(run_name=f"Model_{subset}") as run:
        mlflow.sklearn.log_model(modele, artifact_path=chemin)
        run_id = run.info.run_id 
        print(f"Modèle sauvegardé avec MLflow dans l'expérience {exper_name}, subset : {subset}.")


def charger_modele(chemin):
    """
    Charge un modèle sauvegardé depuis MLflow.
    Arguments:
    - chemin : str : Chemin du modèle sauvegardé.
    Retourne:
    - Le modèle chargé.
    """
    return mlflow.sklearn.load_model(chemin)

def pipeline(chemin_dossier, cible, critere, exper_name="default_experiment"):
    """
    Pipeline complet pour charger les données, entraîner des modèles sur chaque subset, et les sauvegarder.
    Arguments:
    - chemin_dossier : str : Chemin vers le dossier contenant les fichiers CSV.
    - cible : str : Colonne cible pour la classification.
    - critere : str : Colonne pour créer des subsets.
    - exper_name : str : Nom de l'expérience MLflow.
    Retourne:
    - Un dictionnaire avec les modèles entraînés et leurs scores.
    """
    # 1. Chargement des données
    df = load_data(chemin_dossier)

    # 2. Nettoyage des données
    df = clean_data(df)

    # 3. Création de subsets
    subset_sup, subset_inf = subsets(df, critere)

    resultats = {}

    for subset, nom_subset in zip([subset_sup, subset_inf], ["Supérieur", "Inférieur"]):

        X, y, scaler = preprocessing(subset, cible)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        trainer = ModelTrainer()
        best_model, scores = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

        sauvegarder_modele(best_model, f"modele_sklearn_{nom_subset}", exper_name, nom_subset)

        resultats[nom_subset] = {
            'modele': best_model,
            'scores': scores
        }

    return resultats