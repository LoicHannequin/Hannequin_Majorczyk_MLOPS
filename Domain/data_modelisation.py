from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': (RandomForestClassifier(random_state=42), {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }),
            'GradientBoosting': (GradientBoostingClassifier(random_state=42), {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10]
            }),
            'LogisticRegression': (LogisticRegression(max_iter=1000), {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }),
            'SVC': (SVC(probability=True), {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            })
        }
        self.best_model = None
        self.best_score = 0
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Entraîne et évalue plusieurs modèles avec GridSearchCV pour choisir le meilleur.
        Arguments:
        - X_train : ndarray : Données d'entraînement.
        - y_train : ndarray : Labels d'entraînement.
        - X_test : ndarray : Données de test.
        - y_test : ndarray : Labels de test.
        Retourne:
        - Le meilleur modèle entraîné.
        - Les scores associés.
        """
        for name, (model, params) in self.models.items():
            grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=5)
            grid_search.fit(X_train, y_train)
            score = grid_search.best_score_
            print(f"Model: {name}, Best Score: {score:.4f}, Best Params: {grid_search.best_params_}")
            if score > self.best_score:
                self.best_score = score
                self.best_model = grid_search.best_estimator_
        return self.best_model, {"accuracy": self.best_score}

def model_testing(modele, X_test, y_test):
    """
    Évalue le modèle sur les données de test.
    Arguments:
    - modele : modèle sklearn : Le modèle entraîné.
    - X_test : ndarray : Données de test.
    - y_test : ndarray : Labels de test.
    Retourne:
    - Un dictionnaire avec les scores et rapports.
    """
    y_pred = modele.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True), 
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return scores

