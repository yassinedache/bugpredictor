# SVM avec algorithme génétique pour la classification des défauts logiciels
# Version adaptée pour Google Colab

# Installation des dépendances
# !pip install liac-arff scikit-learn matplotlib pandas numpy deap imbalanced-learn

# Importer les bibliothèques
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os
import arff  # liac-arff

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score, make_scorer,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline

from deap import base, creator, tools, algorithms
import joblib
import warnings
from google.colab import drive, files

# Monter Google Drive (à décommenter lors de l'exécution dans Colab)
# drive.mount('/content/drive')

# Fonction pour télécharger les fichiers .arff si nécessaire
def upload_arff_files():
    """Permet à l'utilisateur de télécharger des fichiers .arff."""
    uploaded = files.upload()
    return list(uploaded.keys())

# ============================
# Fonction principale
# ============================
def process_dataset(file_path):
    """Traite un dataset .arff et applique l'algorithme génétique pour optimiser un modèle SVM."""
    print(f"\n📂 Traitement du fichier: {os.path.basename(file_path)}")
    
    # Charger le fichier .arff
    with open(file_path, 'r') as f:
        dataset = arff.load(f)
    
    # Créer un dataframe
    df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    
    # Convertir la variable cible en format numérique
    df['bug'] = df['bug'].apply(lambda x: 1 if str(x).strip().lower() != 'false' else 0)
    
    # Séparer les caractéristiques (X) et la cible (y)
    X = df.drop('bug', axis=1)
    y = df['bug']
    
    # Afficher la distribution des classes avant équilibrage
    class_counts = pd.Series(y).value_counts()
    print(f"\n📊 Distribution des classes avant équilibrage:")
    print(f"   Classe 0 (non-bugs): {class_counts[0]}")
    print(f"   Classe 1 (bugs): {class_counts[1] if 1 in class_counts else 0}")
    print(f"   Ratio bugs/non-bugs: {class_counts[1]/class_counts[0] if 1 in class_counts and class_counts[0] > 0 else 0:.2f}")
    
    # Standardiser les données
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Sélection de caractéristiques basée sur l'information mutuelle
    print("\n🔍 Analyse des caractéristiques par information mutuelle...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranking = np.argsort(mi_scores)[::-1]
    feature_importances = pd.DataFrame({
        'Feature': range(len(mi_scores)),
        'Importance': mi_scores
    }).sort_values('Importance', ascending=False)
    print(f"   Top 10 caractéristiques les plus importantes: {mi_ranking[:10]}")
    
    # Créer des probabilités pour initialiser les features dans l'algorithme génétique
    feature_probs = np.clip(mi_scores / (np.sum(mi_scores) + 1e-10), 0.2, 0.8)
    
    # Utiliser SMOTETomek pour équilibrer les données (combine sur-échantillonnage et sous-échantillonnage)
    print("\n⚖️ Application de SMOTETomek pour équilibrer les données...")
    smote_tomek = SMOTETomek(random_state=42)
    X, y = smote_tomek.fit_resample(X, y)
    
    # Afficher la distribution des classes après équilibrage
    new_class_counts = pd.Series(y).value_counts()
    print(f"   Distribution des classes après équilibrage:")
    print(f"   Classe 0 (non-bugs): {new_class_counts[0]}")
    print(f"   Classe 1 (bugs): {new_class_counts[1]}")
    print(f"   Ratio: {new_class_counts[1]/new_class_counts[0]:.2f}")
    
    # Obtenir le nombre de caractéristiques
    num_features = X.shape[1]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    # ============================
    # Configuration de l'algorithme génétique
    # ============================
    toolbox = base.Toolbox()
    
    # Nettoyer les classes existantes si elles existent déjà
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual
      # Créer les classes pour l'algorithme génétique
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  # Pour F1, précision et rappel
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # Générateurs d'attributs améliorés
    toolbox.register("attr_c", random.uniform, 0.01, 10.0)  # Plage élargie
    toolbox.register("attr_gamma", random.uniform, 0.001, 10.0)  # Plage élargie
    toolbox.register("attr_kernel", random.randint, 0, len(kernels) - 1)
    toolbox.register("attr_degree", random.randint, 2, 5)  # Pour kernel='poly'
    toolbox.register("attr_coef0", random.uniform, 0.0, 10.0)  # Pour 'poly' et 'sigmoid'
    
    # Fonction biaisée pour initialiser les features selon leur importance
    def biased_feature_init():
        features = []
        for i in range(num_features):
            # Utiliser les probabilités basées sur l'information mutuelle
            if random.random() < feature_probs[i]:
                features.append(1)
            else:
                features.append(0)
        return features
    
    # Individu et population avec hyperparamètres additionnels
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_c, toolbox.attr_gamma, toolbox.attr_kernel, 
                      toolbox.attr_degree, toolbox.attr_coef0), n=1)
    
    # Ajouter les features avec biais vers les plus informatives
    toolbox.register("init_features", biased_feature_init)
    toolbox.register("individual_features", tools.initRepeat, list, toolbox.init_features, n=1)
    
    # Fonction pour créer un individu complet
    def create_individual():
        ind = toolbox.individual()
        features = toolbox.individual_features()[0]  # Obtenir les features
        ind.extend(features)  # Ajouter les features à l'individu
        return ind
    
    # Enregistrer la fonction pour créer un individu complet
    toolbox.register("individual_complete", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual_complete)
    
    # Fonction d'évaluation avec multiples objectifs (F1, précision, rappel)
    def evaluate(individual):
        C = max(0.01, individual[0])
        gamma = max(0.001, individual[1])
        kernel = kernels[int(round(individual[2])) % len(kernels)]
        degree = max(2, min(5, int(round(individual[3]))))  # Pour kernel='poly'
        coef0 = max(0.0, individual[4])  # Pour kernels 'poly' et 'sigmoid'
        
        # Les caractéristiques commencent à l'indice 5 maintenant
        feature_mask = individual[5:]
        
        if sum(feature_mask) == 0:
            return -1.0, -1.0, -1.0
            
        selected = [i for i, bit in enumerate(feature_mask) if bit == 1]
        X_sel = X[:, selected]
        
        # Utiliser StratifiedKFold pour une validation croisée équilibrée
        num_folds = 5
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        try:
            # Configuration du SVM avec paramètres additionnels
            clf = SVC(
                C=C, 
                gamma=gamma, 
                kernel=kernel,
                degree=degree if kernel == 'poly' else 3,
                coef0=coef0 if kernel in ['poly', 'sigmoid'] else 0.0,
                class_weight='balanced',
                probability=True
            )
            
            # Collecter les métriques pour chaque fold
            precision_vals = []
            recall_vals = []
            f1_vals = []
            
            for train_idx, test_idx in skf.split(X_sel, y):
                X_train, X_test = X_sel[train_idx], X_sel[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                # Calculer les métriques pour ce fold
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                precision_vals.append(precision)
                recall_vals.append(recall)
                f1_vals.append(f1)
            
            # Retourner la moyenne des métriques sur tous les folds
            return (
                np.mean(f1_vals),
                np.mean(precision_vals),
                np.mean(recall_vals)
            )
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation: {e}")
            return -1.0, -1.0, -1.0
    
    # Opérateurs génétiques
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
      def custom_mutation(individual, indpb):
        # Pour les premiers paramètres (C, gamma, kernel, degree, coef0)
        for i in range(5):
            if random.random() < indpb:
                if i == 0:  # C
                    individual[i] = random.uniform(0.001, 20.0)
                elif i == 1:  # gamma
                    individual[i] = random.uniform(0.0001, 10.0)
                elif i == 2:  # kernel
                    individual[i] = random.randint(0, len(kernels) - 1)
                elif i == 3:  # degree
                    individual[i] = random.randint(2, 5)
                elif i == 4:  # coef0
                    individual[i] = random.uniform(0.0, 10.0)
        
        # Pour les caractéristiques
        for i in range(5, len(individual)):
            if random.random() < indpb:
                individual[i] = 1 - individual[i]  # Inverser le bit
                
        return individual,
    
    toolbox.register("mutate", custom_mutation, indpb=0.05)
    
    # ============================
    # Exécution de l'algorithme génétique
    # ============================
    def run_ga():
        population = toolbox.population(n=50)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
    
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats, verbose=True)
        return population, logbook
      # Exécuter l'algorithme 10 fois et collecter les résultats
    results = []
    all_models = []
    
    for run in range(1, 11):
        print(f"\n================ Exécution {run} ================\n")
        population, logbook = run_ga()
    
        best_ind = tools.selBest(population, 1)[0]
        best_C = max(0.01, best_ind[0])
        best_gamma = max(0.001, best_ind[1])  # Gamma minimum réduit
        best_kernel = kernels[int(best_ind[2]) % len(kernels)]
        best_degree = max(2, min(5, int(round(best_ind[3]))))
        best_coef0 = max(0.0, best_ind[4])
        
        # Les caractéristiques commencent à l'indice 5 maintenant
        selected_features = [i for i, bit in enumerate(best_ind[5:]) if bit == 1]
    
        X_sel = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42, stratify=y)
    
        # Créer un ensemble de modèles SVM pour améliorer la robustesse
        base_models = []
        for i in range(5):  # Créer 5 SVMs de base
            base_svm = SVC(
                C=best_C, 
                gamma=best_gamma, 
                kernel=best_kernel,
                degree=best_degree if best_kernel == 'poly' else 3,
                coef0=best_coef0 if best_kernel in ['poly', 'sigmoid'] else 0.0,
                probability=True,
                class_weight='balanced',
                random_state=42+i
            )
            base_models.append((f'svm_{i}', base_svm))
        
        # Créer un modèle d'ensemble avec vote majoritaire pondéré
        ensemble_model = VotingClassifier(
            estimators=base_models,
            voting='soft'  # Utiliser les probabilités de chaque classifieur
        )
        
        # Entraîner le modèle d'ensemble
        ensemble_model.fit(X_train, y_train)
        pred = ensemble_model.predict(X_test)
        
        # Évaluer sur une gamme de seuils de décision
        if hasattr(ensemble_model, "predict_proba"):
            proba = ensemble_model.predict_proba(X_test)
            
            # Calculer la courbe ROC
            fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            # Afficher la courbe ROC
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taux de faux positifs')
            plt.ylabel('Taux de vrais positifs')
            plt.title('Courbe ROC')
            plt.legend(loc="lower right")
            plt.show()
            
            # Calculer et afficher la courbe de précision-rappel
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, proba[:, 1])
            avg_precision = average_precision_score(y_test, proba[:, 1])
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
                     label=f'Courbe Précision-Rappel (AP = {avg_precision:.2f})')
            plt.xlabel('Rappel')
            plt.ylabel('Précision')
            plt.title('Courbe Précision-Rappel')
            plt.legend(loc="lower left")
            plt.show()
    
        print(f"Meilleur individu: {best_ind}")
        print(f"Caractéristiques sélectionnées ({len(selected_features)}): {selected_features}")
        print(f"Paramètres: C={best_C}, gamma={best_gamma}, kernel={best_kernel}, degree={best_degree}, coef0={best_coef0}")
        print(f"Précision sur l'ensemble de test: {accuracy_score(y_test, pred)}")
        print(classification_report(y_test, pred))
    
        # Afficher la matrice de confusion
        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure(figsize=(10, 8))
        disp.plot(cmap='Blues')
        plt.title('Matrice de confusion')
        plt.show()
    
        # Calculer les métriques
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average='weighted')
        recall = recall_score(y_test, pred, average='weighted')
        f1 = f1_score(y_test, pred, average='weighted')
        
        # Sauvegarder le modèle
        model_info = {
            'model': ensemble_model,
            'features': selected_features,
            'params': {
                'C': best_C,
                'gamma': best_gamma,
                'kernel': best_kernel,
                'degree': best_degree,
                'coef0': best_coef0
            },
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
        all_models.append(model_info)
    
        # Ajouter les résultats
        results.append({
            "run": run,
            "C": best_C,
            "gamma": best_gamma,
            "kernel": best_kernel,
            "degree": best_degree,
            "coef0": best_coef0,
            "features": selected_features,
            "num_features": len(selected_features),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    
    # Afficher le résumé
    results_df = pd.DataFrame(results)
    print("\n📋 Résumé de toutes les exécutions:\n", results_df)
    
    # Sauvegarder le meilleur modèle
    best_run = results_df.loc[results_df['accuracy'].idxmax()]
    best_run_idx = int(best_run['run']) - 1
    
    print(f"\n🏆 Meilleur modèle de l'exécution {best_run['run']} avec précision: {best_run['accuracy']}")
    
    return results_df, best_run

# ============================
# Code principal
# ============================

# Définir comment obtenir les fichiers .arff
def main():
    print("💻 SVM avec algorithme génétique pour la classification des défauts logiciels 💻")
    print("=" * 70)
    
    choice = input("Comment souhaitez-vous fournir les fichiers .arff? (1: Télécharger, 2: Google Drive): ")
    
    if choice == '1':
        print("\nVeuillez télécharger vos fichiers .arff...")
        files_list = upload_arff_files()
        
        for file in files_list:
            process_dataset(file)
            
    elif choice == '2':
        # Monter Google Drive si ce n'est pas déjà fait
        try:
            drive.mount('/content/drive')
        except:
            print("Google Drive est déjà monté.")
        
        dataset_path = input("Entrez le chemin vers votre dossier de datasets dans Google Drive (ex: /content/drive/MyDrive/datasets): ")
        
        if os.path.exists(dataset_path):
            for filename in os.listdir(dataset_path):
                if filename.endswith(".arff"):
                    file_path = os.path.join(dataset_path, filename)
                    process_dataset(file_path)
        else:
            print(f"Le chemin {dataset_path} n'existe pas. Veuillez vérifier et réessayer.")
    else:
        print("Option non valide. Veuillez redémarrer le notebook.")

if __name__ == "__main__":
    main()
