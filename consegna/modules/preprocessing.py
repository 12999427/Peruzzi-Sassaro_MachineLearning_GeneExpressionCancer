import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

class GenePreprocessor:
    def __init__(self, variance_threshold=0.1, n_components=0.95, seed=42):
        self.vt = VarianceThreshold(threshold=variance_threshold) #colonne togli/agiungi
        self.scaler = StandardScaler() # prende i valori di ogni gene e li trasforma in modo che nella colonna abbiano media = 0 e deviazione standard = 1 - i valorei dicentano piccoli e comparabili tra loro
        self.pca = PCA(n_components=n_components, random_state=seed) # comprime le dimensioni, creandone nuove che rappresentano più significativamente
        self.le = LabelEncoder() # da etichette a indici dei gruppi
        self.seed = seed


    #c'è uno studio dei calcoli della varianza, e dopo applicata, e così anche per altri fattori matematici
    def fit_transform(self, X, y=None): #applica la pipeline completa, fittando, ovvero calcolando i parametri necessari - lo si fa in training
        X_vt = self.vt.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_vt)
        X_pca = self.pca.fit_transform(X_scaled)
        
        if y is not None:
            y_enc = self.le.fit_transform(y) #
            return X_pca, y_enc
        
        return X_pca


    #applica la trasformazione anche ai nuovi/futuri dati o a quelli da testare
    #qua non c'è fit perchè usa i valori calcolati durante la fase di training (e conservati negli oggetto dichiarati nel costruttore)
    #per applicarli nei nuovi dati
    #in modo tale da valutare le stesse colonne e stessi tipi di dati, normalizzati alla stessa maniera ecc
    def transform(self, X, y=None): # applica la pipeline su dati da testare
        X_vt = self.vt.transform(X)
        X_scaled = self.scaler.transform(X_vt)
        X_pca = self.pca.transform(X_scaled)
        
        if y is not None:
            y_enc = self.le.transform(y)
            return X_pca, y_enc
        
        return X_pca
