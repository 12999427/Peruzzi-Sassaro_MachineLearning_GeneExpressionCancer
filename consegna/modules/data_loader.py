import pandas as pd
import os

def load_cancer_data(data_path='../archive/data.csv', labels_path='../archive/labels.csv'):
    """
    Carica il dataset RNA-seq e le relative etichette.
    Nota: i percorsi sono relativi alla posizione dei notebook.
    """
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        # Prova percorso alternativo (se eseguito dalla root)
        data_path = 'archive/data.csv'
        labels_path = 'archive/labels.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError("File non trovati. Verifica i percorsi di data.csv e labels.csv")
    
    X = pd.read_csv(data_path, index_col=0)
    y = pd.read_csv(labels_path, index_col=0)
    
    return X, y['Class']
