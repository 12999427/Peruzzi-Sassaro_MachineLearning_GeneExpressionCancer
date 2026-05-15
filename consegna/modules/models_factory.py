import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import GridSearchCV

def get_supervised_model(name, seed=42):
    """
    Ritorna un modello supervisionato con i parametri di base o ottimali.
    """
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5, 
            class_weight='balanced', random_state=seed, n_jobs=-1
        ),
        'svm': SVC(
            C=10, kernel='rbf', probability=True, 
            class_weight='balanced', random_state=seed
        ),
        'logistic_regression': LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=seed
        )
    }
    return models.get(name)

def tune_hyperparameters(name, X, y, seed=42):
    """
    Esegue GridSearchCV per trovare i parametri migliori.
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    }
    
    base_model = get_supervised_model(name, seed=seed)
    grid_search = GridSearchCV(base_model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

def get_unsupervised_model(name, n_clusters=5, seed=42):
    """
    Ritorna un modello di clustering.
    """
    models = {
        'kmeans': KMeans(n_clusters=n_clusters, n_init=20, random_state=seed),
        'spectral': SpectralClustering(
            n_clusters=n_clusters, affinity='nearest_neighbors', 
            n_neighbors=10, random_state=seed, n_jobs=-1
        )
    }
    return models.get(name)

def save_model(model, name, folder='models'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, f"{name}.joblib")
    joblib.dump(model, path)
    return path

def save_results(results, name, folder='results'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    return path
