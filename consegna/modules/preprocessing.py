import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

class GenePreprocessor:
    def __init__(self, variance_threshold=0.1, n_components=0.95, seed=42):
        self.vt = VarianceThreshold(threshold=variance_threshold)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.le = LabelEncoder()
        self.seed = seed

    def fit_transform(self, X, y=None):
        """
        Applica la pipeline completa: VarianceThreshold -> Scaling -> PCA.
        Se y è fornito, applica anche LabelEncoder.
        """
        X_vt = self.vt.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_vt)
        X_pca = self.pca.fit_transform(X_scaled)
        
        if y is not None:
            y_enc = self.le.fit_transform(y)
            return X_pca, y_enc
        
        return X_pca

    def transform(self, X, y=None):
        """
        Applica le trasformazioni già fittate.
        """
        X_vt = self.vt.transform(X)
        X_scaled = self.scaler.transform(X_vt)
        X_pca = self.pca.transform(X_scaled)
        
        if y is not None:
            y_enc = self.le.transform(y)
            return X_pca, y_enc
        
        return X_pca
