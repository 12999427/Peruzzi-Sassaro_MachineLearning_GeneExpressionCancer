import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import GridSearchCV

def get_supervised_model(name, seed=42): # i parametri sono di base, poi sotto li tuna
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5, 
            class_weight='balanced', random_state=seed, n_jobs=-1 #n_jobs=-1 per usare tutti i core disponibili, class_weight='balanced' per bilanciare le classi in caso di una sovrerappresentazione di una classe rispetto all'altra
        ),
        'svm': SVC(
            C=10, kernel='rbf', probability=True, #ovvero non usare vero/falso per classificare, ma un flot di probabilità
            class_weight='balanced', random_state=seed
        ),
        'logistic_regression': LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=seed
        )
    }
    return models.get(name)

def tune_hyperparameters(name, X, y, seed=42): # tuning con GridSearchCV per trovare i parametri migliori.
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 300], #numero di alberi nella foresta
            'max_depth': [None, 10, 20], #profondità massima degli alberi
            'min_samples_split': [2, 5] #numero minimo di campioni per "giustificare" una divisione
        },
        'svm': {
            'C': [0.1, 1, 10], #penalità per errori
            'kernel': ['linear', 'rbf'] #linear per creare confini lineari, rbf per confini più complessi
        },
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    }
    
    base_model = get_supervised_model(name, seed=seed) #modello base
    grid_search = GridSearchCV(base_model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1) #prova tutte le combinazioni specificate. Effettua cross validation per verificare i risultati
    
    #ora i parametri sono ottimali, lo allena qui
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

def get_unsupervised_model(name, n_clusters=5, seed=42):
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

"""
(Cross-Validation):
       * Usa le parti 1 e 2 per allenarsi e la 3 per testare.
       * Usa le parti 1 e 3 per allenarsi e la 2 per testare.
       * Usa le parti 2 e 3 per allenarsi e la 1 per testare.
       * Fa la media dei risultati.
"""