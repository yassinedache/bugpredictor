#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K-Nearest Neighbors (KNN) avec optimisation par algorithme génétique pour la prédiction de défauts logiciels.
Ce module fournit des fonctions pour l'entraînement, l'optimisation et l'évaluation d'un modèle KNN 
pour détecter les défauts dans le code logiciel à partir de métriques de code.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, precision_recall_curve, auc, roc_curve,
                            confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN, SMOTETomek
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Import des fonctions d'optimisation par algorithme génétique
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous

# Import des fonctions d'utilité du projet
from utils.metrics import print_metrics
from utils.cv_utils import evaluate_model
from data.preprocess import preprocess_data

# Constantes
RANDOM_STATE = 42
DATA_PATH = os.path.join('..', 'datasets', 'SDP-dataset')


def load_arff_data(file_path):
    """
    Charge un fichier ARFF et retourne les données sous forme de DataFrame Pandas
    et la variable cible (défaut/non-défaut).
    
    Args:
        file_path (str): Chemin vers le fichier ARFF
        
    Returns:
        tuple: (X, y) - features et variable cible
    """
    try:
        # Essayer de charger un fichier CSV d'abord (pour la compatibilité)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, sep=';')
            y = df.iloc[:, -1].values  # La dernière colonne est généralement la cible
            X = df.iloc[:, :-1].values
            return X, y
        
        # Sinon charger le fichier ARFF
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # Convertir les types bytes en string (si nécessaire)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')
        
        # La dernière colonne est généralement la cible (défaut ou non)
        target_col = df.columns[-1]
        y = (df[target_col] == 'Y').astype(int).values  # Convertir en binaire (0/1)
        X = df.drop(columns=[target_col]).values
        
        return X, y
    
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path}: {e}")
        return None, None


def handle_class_imbalance(X, y, method='smote', sampling_strategy=0.5):
    """
    Gère le déséquilibre des classes en utilisant différentes méthodes de rééchantillonnage.
    
    Args:
        X (array): Features
        y (array): Variable cible
        method (str): Méthode de rééchantillonnage ('smote', 'random_under', 'smoteenn', 'smotetomek')
        sampling_strategy (float/dict): Stratégie de rééchantillonnage
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    if method == 'smote':
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    elif method == 'random_under':
        resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    elif method == 'smoteenn':
        resampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    elif method == 'smotetomek':
        resampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    else:
        print(f"Méthode {method} non reconnue. Utilisation des données originales.")
        return X, y
    
    # Appliquer le rééchantillonnage
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    # Afficher les informations sur le rééchantillonnage
    print(f"Rééchantillonnage avec {method}:")
    print(f"  Classes originales: {np.bincount(y)}")
    print(f"  Classes après rééchantillonnage: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled


def run_grid_search_knn(X_train, y_train, X_test, y_test, cv=5):
    """
    Exécute une recherche par grille pour trouver les meilleurs hyperparamètres du modèle KNN.
    
    Args:
        X_train (array): Données d'entraînement
        y_train (array): Étiquettes d'entraînement
        X_test (array): Données de test
        y_test (array): Étiquettes de test
        cv (int): Nombre de plis pour la validation croisée
        
    Returns:
        dict: Résultats de la recherche par grille
    """
    # Créer une pipeline pour prétraitement + KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feat_select', SelectKBest(score_func=mutual_info_classif)),
        ('classifier', KNeighborsClassifier())
    ])
    
    # Paramètres à optimiser
    param_grid = {
        'feat_select__k': [5, 10, 'all'],  # Nombre de features à sélectionner
        'classifier__n_neighbors': list(range(1, 31, 2)),  # Nombre de voisins impair de 1 à 30
        'classifier__weights': ['uniform', 'distance'],  # Poids des voisins
        'classifier__p': [1, 2],  # Distance de Minkowski (1=Manhattan, 2=Euclidienne)
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Algorithme de recherche
    }
    
    # Créer le stratified K-fold pour la validation croisée
    cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    # Métrique principale: F1-score (important pour les classes déséquilibrées)
    grid_search = GridSearchCV(
        pipeline, param_grid, scoring='f1_macro', cv=cv_folds, 
        n_jobs=-1, verbose=1, return_train_score=True
    )
    
    # Exécuter la recherche par grille
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Afficher les résultats
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score F1: {grid_search.best_score_:.4f}")
    print(f"Temps de recherche: {search_time:.2f} secondes")
    
    # Évaluer sur l'ensemble de test
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nPerformances sur l'ensemble de test:")
    print_metrics(best_model, X_test, y_test)
    
    # Sauvegarder le meilleur modèle
    joblib.dump(best_model, 'best_knn_model_grid_search.pkl')
    
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'search_time': search_time
    }


def run_genetic_algorithm_knn(X_train, y_train, X_test, y_test, cv=5, 
                             population_size=50, generations=20, 
                             scoring='f1_macro'):
    """
    Optimise les hyperparamètres du modèle KNN en utilisant un algorithme génétique.
    
    Args:
        X_train, y_train, X_test, y_test: Données d'entraînement et de test
        cv: Nombre de plis pour la validation croisée
        population_size: Taille de la population pour l'algorithme génétique
        generations: Nombre de générations
        scoring: Métrique d'évaluation ('f1_macro', 'roc_auc', etc.)
        
    Returns:
        dict: Modèle optimisé et résultats
    """
    # Définir l'espace de recherche
    param_grid = {
        'n_neighbors': Integer(1, 50),  # Espace de recherche plus large
        'weights': Categorical(['uniform', 'distance']),
        'p': Categorical([1, 2]),  # 1=Manhattan, 2=Euclidean
        'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': Integer(10, 100)  # Paramètre pour 'ball_tree' et 'kd_tree'
    }
    
    # Prétraitement avant GA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Création du modèle KNN
    knn = KNeighborsClassifier()
    
    # Configuration de l'algorithme génétique
    ga_search = GASearchCV(
        estimator=knn,
        cv=cv,
        param_grid=param_grid,
        scoring=scoring,
        population_size=population_size,
        generations=generations,
        n_jobs=-1,
        verbose=True,
        crossover_probability=0.9,
        mutation_probability=0.1,
        tournament_size=3,
        elitism=True,
        random_state=RANDOM_STATE
    )
    
    # Exécution de l'algorithme génétique
    print("Démarrage de l'optimisation par algorithme génétique...")
    start_time = time.time()
    ga_search.fit(X_train_scaled, y_train)
    search_time = time.time() - start_time
    
    # Afficher les résultats
    print(f"\nMeilleurs paramètres trouvés: {ga_search.best_params_}")
    print(f"Meilleur score {scoring}: {ga_search.best_score_:.4f}")
    print(f"Temps d'optimisation: {search_time:.2f} secondes")
    
    # Créer le modèle final avec les meilleurs paramètres
    best_knn = KNeighborsClassifier(**ga_search.best_params_)
    best_knn.fit(X_train_scaled, y_train)
    
    # Évaluer sur l'ensemble de test
    print("\nPerformances sur l'ensemble de test:")
    print_metrics(best_knn, X_test_scaled, y_test)
    
    # Sauvegarder le meilleur modèle
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', best_knn)
    ])
    joblib.dump(pipeline, 'best_knn_model_ga.pkl')
    
    return {
        'model': pipeline,
        'best_params': ga_search.best_params_,
        'best_score': ga_search.best_score_,
        'ga_results': ga_search.history_,
        'search_time': search_time
    }


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Génère une courbe d'apprentissage pour visualiser la relation entre la taille d'entraînement,
    les scores d'entraînement et de validation.
    """
    from sklearn.model_selection import learning_curve
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Score F1")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='f1_macro', shuffle=True, random_state=RANDOM_STATE)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Tracer les scores d'entraînement
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Score d'entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Score de validation croisée")

    plt.legend(loc="best")
    plt.savefig('knn_learning_curve.png')
    plt.close()
    
    return plt


def plot_cross_project_predictions(models, datasets, save_path='cross_project_results.png'):
    """
    Évalue les performances du modèle sur différents projets
    et génère une visualisation pour la comparaison entre projets.
    """
    results = []
    
    for model_name, model in models.items():
        project_scores = []
        
        for dataset_name, dataset_path in datasets.items():
            X, y = load_arff_data(dataset_path)
            
            if X is not None and y is not None:
                # Évaluer le modèle sur ce dataset
                try:
                    y_pred = model.predict(X)
                    f1 = f1_score(y, y_pred, average='macro')
                    auc_score = roc_auc_score(y, model.predict_proba(X)[:, 1]) if hasattr(model, 'predict_proba') else 0
                    
                    project_scores.append({
                        'dataset': dataset_name,
                        'f1_score': f1,
                        'auc_score': auc_score
                    })
                except Exception as e:
                    print(f"Erreur lors de l'évaluation sur {dataset_name}: {e}")
        
        # Ajouter les scores au résultat global
        if project_scores:
            results.append({
                'model': model_name,
                'scores': project_scores
            })
    
    # Créer un DataFrame pour faciliter la visualisation
    rows = []
    for result in results:
        for score in result['scores']:
            rows.append({
                'Model': result['model'],
                'Project': score['dataset'],
                'F1 Score': score['f1_score'],
                'AUC': score['auc_score']
            })
    
    df = pd.DataFrame(rows)
    
    # Visualiser les résultats
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Model', y='F1 Score', data=df)
    plt.title('F1 Scores par Modèle (Cross-Project)')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Model', y='AUC', data=df)
    plt.title('AUC Scores par Modèle (Cross-Project)')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return df


def main():
    """
    Fonction principale qui exécute le pipeline complet de prédiction de défauts logiciels.
    """
    print("=== Prédiction de Défauts Logiciels avec KNN ===")
    
    # Charger les données (exemple avec le dataset ant-1.7.arff)
    dataset_path = os.path.join(DATA_PATH, 'ant-1.7.arff')
    print(f"Chargement des données depuis {dataset_path}")
    
    X, y = load_arff_data(dataset_path)
    
    if X is None or y is None:
        print("Erreur lors du chargement des données. Arrêt du programme.")
        return
    
    print(f"Forme des données chargées: X={X.shape}, y={np.bincount(y)}")
    
    # Prétraitement des données
    X_selected, selector, scaler, imputer = preprocess_data(X, y)
    
    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Gérer le déséquilibre des classes
    print("\n=== Gestion du déséquilibre des classes ===")
    X_train_resampled, y_train_resampled = handle_class_imbalance(
        X_train, y_train, method='smote', sampling_strategy=0.7
    )
    
    # 1. Optimisation des hyperparamètres avec GridSearchCV
    print("\n=== Optimisation avec GridSearchCV ===")
    grid_results = run_grid_search_knn(
        X_train_resampled, y_train_resampled, X_test, y_test, cv=5
    )
    
    # 2. Optimisation des hyperparamètres avec l'algorithme génétique
    print("\n=== Optimisation avec Algorithme Génétique ===")
    ga_results = run_genetic_algorithm_knn(
        X_train_resampled, y_train_resampled, X_test, y_test, 
        cv=5, population_size=50, generations=20, scoring='f1_macro'
    )
    
    # Comparer les performances des différents modèles
    print("\n=== Comparaison des Modèles ===")
    models = {
        'KNN_GridSearch': grid_results['model'],
        'KNN_GA': ga_results['model']
    }
    
    print("F1-score sur l'ensemble de test:")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"  {name}: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    # Générer la courbe d'apprentissage
    print("\n=== Génération de la courbe d'apprentissage ===")
    best_model = models['KNN_GA'] if ga_results['best_score'] > grid_results['best_score'] else models['KNN_GridSearch']
    plot_learning_curve(best_model, 'Courbe d\'apprentissage - KNN', X_selected, y, cv=5)
    
    # Évaluer les performances du modèle sur d'autres projets (cross-project)
    print("\n=== Évaluation Cross-Project ===")
    datasets = {
        'ant-1.7': os.path.join(DATA_PATH, 'ant-1.7.arff'),
        'camel-1.6': os.path.join(DATA_PATH, 'camel-1.6.arff'),
        'jedit-4.3': os.path.join(DATA_PATH, 'jedit-4.3.arff'),
        'lucene-2.4': os.path.join(DATA_PATH, 'lucene-2.4.arff'),
        'xerces-1.4': os.path.join(DATA_PATH, 'xerces-1.4.arff')
    }
    cross_project_df = plot_cross_project_predictions(models, datasets)
    
    print("\nRésultats sauvegardés. Le meilleur modèle est disponible dans 'best_knn_model_ga.pkl'")
    

if __name__ == "__main__":
    main()