import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os
import arff

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek

from deap import base, creator, tools, algorithms
import joblib


# ============================
# Load and preprocess dataset
# ============================
for filename in os.listdir("C:/Users/TRETEC/Documents/dataset"):
    if filename.endswith(".arff"):
        file_path = os.path.join("C:/Users/TRETEC/Documents/dataset", filename)
        print(f"\n📂 Processing file: {filename}")

        with open(file_path, 'r') as f:
            dataset = arff.load(f)
        
        df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
        df['bug'] = df['bug'].apply(lambda x: 1 if str(x).strip().lower() != 'false' else 0)
        
        X = df.drop('bug', axis=1)
        y = df['bug']
        
        # Check class balance before SMOTE
        class_counts = np.bincount(y.astype(int))
        print(f"Class distribution before resampling: {class_counts}")
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        
        # Check class balance after SMOTE
        class_counts_after = np.bincount(y.astype(int))
        print(f"Class distribution after resampling: {class_counts_after}")
        
        # Feature count
        num_features = X.shape[1]
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        
        # ============================
        # Genetic Algorithm Setup
        # ============================
        toolbox = base.Toolbox()
        
        if "FitnessMax" in creator.__dict__:
            del creator.FitnessMax
        if "Individual" in creator.__dict__:
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Attribute generators - Expanded parameter ranges
        toolbox.register("attr_c", random.uniform, 0.001, 10)        # Wider range for C
        toolbox.register("attr_gamma", random.uniform, 0.001, 10)    # Include smaller gamma values
        toolbox.register("attr_kernel", random.randint, 0, len(kernels) - 1)
        toolbox.register("attr_bool", random.randint, 0, 1)
        
        # Individual and population
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_c, toolbox.attr_gamma, toolbox.attr_kernel) + (toolbox.attr_bool,) * num_features,
                         n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Evaluation function with improved scoring
        def evaluate(individual):
            C = max(0.001, individual[0])
            gamma = max(0.001, individual[1])
            kernel = kernels[int(round(individual[2])) % len(kernels)]
            feature_mask = individual[3:]
        
            if sum(feature_mask) == 0:
                return -1.0,
            
            selected = [i for i, bit in enumerate(feature_mask) if bit == 1]
            X_sel = X[:, selected]
            num_folds = 5
            
            # Use StratifiedKFold instead of regular KFold
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
            
            try:
                # Dynamic class weighting
                class_counts = np.bincount(y.astype(int))
                weight_ratio = max(class_counts) / min(class_counts)
                class_weights = {0: 1, 1: weight_ratio} if class_counts[0] > class_counts[1] else {1: 1, 0: weight_ratio}
                
                clf = SVC(C=C, gamma=gamma, kernel=kernel, class_weight='balanced', probability=True)
                
                # Calculate metrics
                accuracy = cross_val_score(clf, X_sel, y, cv=skf, scoring='accuracy').mean()
                f1 = cross_val_score(clf, X_sel, y, cv=skf, scoring='f1').mean()
                precision = cross_val_score(clf, X_sel, y, cv=skf, scoring='precision').mean()
                recall = cross_val_score(clf, X_sel, y, cv=skf, scoring='recall').mean()
                
                # Calculate AUC-ROC if possible
                try:
                    y_prob = cross_val_predict(clf, X_sel, y, cv=skf, method='predict_proba')
                    auc_roc = roc_auc_score(y, y_prob[:, 1])
                    # Weighted combination with emphasis on F1 score and recall
                    score = 0.6 * f1 + 0.1 * accuracy + 0.1 * precision + 0.1 * recall + 0.1 * auc_roc
                except:
                    # If AUC-ROC fails, use original metrics with modified weights
                    score = 0.6 * f1 + 0.1 * accuracy + 0.1 * precision + 0.2 * recall
            except Exception as e:
                print(f"Error in evaluation: {e}")
                score = -1
                
            return score,
        
        # Operators
        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", tools.cxTwoPoint)
        
        # Improved mutation operator with separate rates for parameters and features
        def custom_mutation(individual, indpb_params=0.1, indpb_features=0.05):
            # Mutate SVM parameters with higher probability
            for i in range(3):
                if random.random() < indpb_params:
                    if i == 0:  # C parameter
                        individual[i] = random.uniform(0.001, 10)
                    elif i == 1:  # gamma parameter
                        individual[i] = random.uniform(0.001, 10)
                    elif i == 2:  # kernel parameter
                        individual[i] = random.randint(0, len(kernels) - 1)
                        
            # Mutate feature selection with different probability
            for i in range(3, len(individual)):
                if random.random() < indpb_features:
                    individual[i] = 1 - individual[i]
            return individual,
        
        toolbox.register("mutate", custom_mutation, indpb_params=0.1, indpb_features=0.05)
        
        # ============================
        # Run the GA
        # ============================
        def main():
            # Increased population size for better exploration
            population = toolbox.population(n=80)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
        
            # Adjusted genetic algorithm parameters
            population, logbook = algorithms.eaSimple(
                population, toolbox, 
                cxpb=0.6,        # Increased crossover probability
                mutpb=0.3,       # Increased mutation probability
                ngen=25,         # Increased generations
                stats=stats, 
                verbose=True
            )
            return population, logbook
        
        if __name__ == "__main__":
            results = []
            all_selected_features = []
            
            for run in range(1, 11):
                print(f"\n================ Run {run} ================\n")
                population, logbook = main()
        
                best_ind = tools.selBest(population, 1)[0]
                best_C = max(0.001, best_ind[0])
                best_gamma = max(0.001, best_ind[1])
                best_kernel = kernels[int(best_ind[2]) % len(kernels)]
                selected_features = [i for i, bit in enumerate(best_ind[3:]) if bit == 1]
                
                # Track selected features
                all_selected_features.append(selected_features)
        
                X_sel = X[:, selected_features]
                X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)
        
                # Final model with probability=True for AUC calculation
                final_model = SVC(C=best_C, gamma=best_gamma, kernel=best_kernel, probability=True)
                final_model.fit(X_train, y_train)
                pred = final_model.predict(X_test)
                
                # For ROC-AUC calculation
                pred_proba = final_model.predict_proba(X_test)
        
                print(f"Best individual: {best_ind}")
                print(f"Selected features ({len(selected_features)}): {selected_features}")
                print(f"Best C: {best_C:.6f}, Best gamma: {best_gamma:.6f}, Best kernel: {best_kernel}")
                print(f"Test Accuracy: {final_model.score(X_test, y_test)}")
                print(classification_report(y_test, pred))
        
                cm = confusion_matrix(y_test, pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues')
                plt.title(f'Confusion Matrix - Run {run}')
                plt.savefig(f'confusion_matrix_run_{run}.png')
                plt.show()
        
                # Calculate all metrics
                accuracy = accuracy_score(y_test, pred)
                precision = precision_score(y_test, pred, average='weighted')
                recall = recall_score(y_test, pred, average='weighted')
                f1 = f1_score(y_test, pred, average='weighted')
                
                # Calculate AUC-ROC if possible
                auc_roc = None
                try:
                    if len(np.unique(y_test)) > 1:
                        auc_roc = roc_auc_score(y_test, pred_proba[:, 1])
                except:
                    print("Could not calculate AUC-ROC")
        
                # Save results
                run_results = {
                    "run": run,
                    "C": best_C,
                    "gamma": best_gamma,
                    "kernel": best_kernel,
                    "features": selected_features,
                    "num_features": len(selected_features),
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
                
                if auc_roc is not None:
                    run_results["auc_roc"] = auc_roc
                    
                results.append(run_results)
                
                # Save each model
                model_filename = f'svm_model_run_{run}.pkl'
                joblib.dump(final_model, model_filename)
                print(f"Model saved as {model_filename}")
        
            # Analyze results
            results_df = pd.DataFrame(results)
            print("\n📋 Summary of all runs:\n", results_df)
            
            # Get best model based on F1 score
            best_f1_idx = results_df['f1'].idxmax()
            best_run = results_df.iloc[best_f1_idx]
            print(f"\n🏆 Best model (Run {best_run['run']}):")
            print(f"   F1 Score: {best_run['f1']:.4f}")
            print(f"   Accuracy: {best_run['accuracy']:.4f}")
            print(f"   Precision: {best_run['precision']:.4f}")
            print(f"   Recall: {best_run['recall']:.4f}")
            if 'auc_roc' in best_run:
                print(f"   AUC-ROC: {best_run['auc_roc']:.4f}")
            print(f"   Parameters: C={best_run['C']:.6f}, gamma={best_run['gamma']:.6f}, kernel={best_run['kernel']}")
            print(f"   Features: {best_run['num_features']} selected")
            
            # Feature importance analysis
            feature_importance = np.zeros(num_features)
            for features in all_selected_features:
                for feature in features:
                    feature_importance[feature] += 1
                    
            feature_importance = feature_importance / len(all_selected_features)  # Normalize
            important_features = np.argsort(feature_importance)[::-1]
            
            print("\n📊 Feature Importance Analysis:")
            print(f"Most important features (top 10): {important_features[:10]}")
            
            # Create ensemble of top 3 models
            print("\n🔄 Creating ensemble model from top 3 models...")
            try:
                top3_indices = results_df['f1'].nlargest(3).index
                ensemble_models = []
                
                for idx in top3_indices:
                    run_num = results_df.iloc[idx]['run']
                    model_path = f'svm_model_run_{run_num}.pkl'
                    model = joblib.load(model_path)
                    ensemble_models.append(model)
                    
                # Save the ensemble info
                ensemble_info = {
                    'models': [f'svm_model_run_{results_df.iloc[idx]["run"]}.pkl' for idx in top3_indices],
                    'feature_sets': [results_df.iloc[idx]['features'] for idx in top3_indices]
                }
                
                joblib.dump(ensemble_info, 'svm_ensemble_info.pkl')
                print("Ensemble information saved as 'svm_ensemble_info.pkl'")
                
            except Exception as e:
                print(f"Error creating ensemble: {e}")
