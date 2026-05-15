# Progetto ML Cancer Genomes
Questo progetto utilizza tecniche di Machine Learning (Supervisionato e Non Supervisionato) per classificare tipi di tumore basandosi su dati di espressione genica (RNA-seq).

## Struttura del Progetto
- `modules/`: Contiene la logica riutilizzabile (caricamento, preprocessing, modelli, grafici).
- `data_processed/`: Contiene i dati dopo lo scaling e la PCA.
- `models/`: Contiene i modelli allenati salvati in formato `.joblib`.
- `results/`: Contiene i risultati delle performance in formato `.json`.

## Notebooks
1. `01_eda_preprocessing.ipynb`: Caricamento e preparazione dei dati.
2. `02_supervised.ipynb`: Addestramento e valutazione di SVM, Random Forest e Logistic Regression.
3. `03_unsupervised.ipynb`: Clustering con KMeans e Spectral Clustering.
4. `04_summary_sensitivity.ipynb`: Riepilogo finale e analisi di sensibilità della PCA.

## Installazione
0.  python -m venv VE
    source VE/bin/activate

1. Installa le dipendenze:
   pip install -r requirements.txt
2. Installa https://www.kaggle.com/datasets/waalbannyantudre/gene-expression-cancer-rna-seq-donated-on-682016 e metti "archive" a fianco a "consegna" - non dentro
3. Esegui i notebook in ordine
