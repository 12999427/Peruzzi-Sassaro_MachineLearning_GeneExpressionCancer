import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D

def plot_expression_distribution(X, n_genes=50):
    """
    Mostra la distribuzione dei valori di espressione per un campione di geni.
    """
    plt.figure(figsize=(12, 6))
    # Prendiamo un campione casuale di geni se sono troppi
    sample_indices = np.random.choice(X.shape[1], min(n_genes, X.shape[1]), replace=False)
    X_sample = X.iloc[:, sample_indices] if hasattr(X, 'iloc') else X[:, sample_indices]
    
    sns.boxplot(data=X_sample)
    plt.title(f"Distribuzione dei valori di espressione ({n_genes} geni a campione)")
    plt.xticks([]) # Nascondiamo i nomi dei geni se sono troppi
    plt.ylabel("Valore Espressione")
    plt.show()

def plot_pca_2d(X_pca, y, le_classes, title="PCA 2D"):
    plt.figure(figsize=(10, 7))
    unique_classes = np.unique(y)
    for cls_idx in unique_classes:
        mask = (y == cls_idx)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=le_classes[cls_idx], alpha=0.7)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Tumori")
    plt.show()

def plot_pca_3d(X_pca, y, le_classes, title="PCA 3D"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_classes = np.unique(y)
    for cls_idx in unique_classes:
        mask = (y == cls_idx)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], label=le_classes[cls_idx], alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(title="Tumori")
    plt.show()

def plot_umap_2d(X_umap, y, le_classes, title="UMAP 2D"):
    plt.figure(figsize=(10, 7))
    unique_classes = np.unique(y)
    for cls_idx in unique_classes:
        mask = (y == cls_idx)
        plt.scatter(X_umap[mask, 0], X_umap[mask, 1], label=le_classes[cls_idx], alpha=0.7)
    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Tumori")
    plt.show()

def plot_umap_3d(X_umap, y, le_classes, title="UMAP 3D"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_classes = np.unique(y)
    for cls_idx in unique_classes:
        mask = (y == cls_idx)
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], X_umap[mask, 2], label=le_classes[cls_idx], alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    ax.legend(title="Tumori")
    plt.show()

def plot_cluster_comparison(X_pca, y_true, y_pred, le_classes, model_name=""):
    """
    Grafico a due pannelli: Classi Reali vs Cluster Calcolati.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Classi Reali
    unique_classes = np.unique(y_true)
    for cls_idx in unique_classes:
        mask = (y_true == cls_idx)
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=le_classes[cls_idx], alpha=0.7)
    
    ax1.set_title("Classi Reali (Ground Truth)")
    ax1.legend(title="Tumori")

    # Plot Cluster Calcolati
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='Set2', alpha=0.7)
    ax2.set_title(f"Cluster Trovati ({model_name})")
    ax2.legend(*scatter2.legend_elements(), title="Cluster ID")

    plt.tight_layout()
    plt.show()

def plot_cluster_comparison_3d(X_pca, y_true, y_pred, le_classes, model_name=""):
    """
    Grafico 3D a due pannelli: Classi Reali vs Cluster Calcolati.
    """
    fig = plt.figure(figsize=(18, 8))
    
    # Classi Reali
    ax1 = fig.add_subplot(121, projection='3d')
    unique_classes = np.unique(y_true)
    for cls_idx in unique_classes:
        mask = (y_true == cls_idx)
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], label=le_classes[cls_idx], alpha=0.7)
    ax1.set_title("Classi Reali (Ground Truth)")
    ax1.legend(title="Tumori")

    # Cluster Calcolati
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_pred, cmap='Set2', alpha=0.7)
    ax2.set_title(f"Cluster Trovati ({model_name})")
    ax2.legend(*scatter2.legend_elements(), title="Cluster ID")

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(title)
    plt.show()

def plot_contingency_matrix(y_true, y_pred, le_classes, title="Matrice di Contingenza (Classi vs Cluster)"):
    """
    Mostra una heatmap che mette in relazione le classi reali con i cluster trovati.
    Utile per capire quale cluster corrisponde a quale tumore.
    """
    import pandas as pd
    data = pd.DataFrame({'Real': [le_classes[i] for i in y_true], 'Cluster': y_pred})
    contingency = pd.crosstab(data['Real'], data['Cluster'])
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(title)
    plt.ylabel("Tumore Reale")
    plt.xlabel("ID Cluster Calcolato")
    plt.show()

def plot_pca_variance(pca):
    exp_var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(exp_var, marker='o', linestyle='-', color='b')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Varianza')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Spiegata Cumulativa')
    plt.title('Analisi Varianza PCA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
