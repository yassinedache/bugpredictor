# --- Install requirements ---
!pip install imblearn deap scipy openpyxl

# --- Imports ---
import random
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE
from deap import base, creator, tools, algorithms
from scipy.io import arff
from google.colab import files
from io import StringIO

# --- 1. Upload ARFF dataset ---
print("📁 Upload your .arff file")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
content = uploaded[filename].decode('utf-8')
data, meta = arff.loadarff(StringIO(content))
df = pd.DataFrame(data)

# --- 2. Prepare dataset ---
target_col = 'bug'  # Adjust if needed

# Convert byte targets to int
if df[target_col].dtype == object or df[target_col].dtype.name == 'bytes':
    df[target_col] = df[target_col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    df[target_col] = df[target_col].apply(lambda x: 1 if x in ['Y', 'yes', 'true', 'True', 1] else 0)

# Check target validity
if df[target_col].nunique() < 2:
    raise ValueError(f"❌ Target '{target_col}' has only one class.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# --- 3. GA Setup for Both Feature Selection and Hyperparameter Tuning ---
NUM_FEATURES = X_resampled.shape[1]

# Reset DEAP creator
if 'FitnessMulti' in creator.__dict__:
    del creator.FitnessMulti
if 'Individual' in creator.__dict__:
    del creator.Individual

# GA: Maximize F1 and Recall
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Hyperparameter ranges
K_RANGE = (1, 21)  # n_neighbors: 1 to 20
WEIGHTS_OPTIONS = ['uniform', 'distance']
METRIC_OPTIONS = ['euclidean', 'manhattan', 'minkowski']

toolbox = base.Toolbox()

def create_individual():
    """Create an individual representing both feature selection and hyperparameters"""
    # Feature selection (binary array)
    features = [random.randint(0, 1) for _ in range(NUM_FEATURES)]
    # Hyperparameters
    k = random.randint(K_RANGE[0], K_RANGE[1] - 1)
    weights = random.choice(WEIGHTS_OPTIONS)
    metric = random.choice(METRIC_OPTIONS)
    return features + [k, weights, metric]

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_fitness(individual):
    # Split individual into features and hyperparameters
    feature_mask = individual[:NUM_FEATURES]
    k, weights, metric = individual[NUM_FEATURES:]
    
    selected_indices = [i for i in range(NUM_FEATURES) if feature_mask[i] == 1]
    if len(selected_indices) == 0:
        return 0.0, 0.0
    
    X_selected = X_resampled[:, selected_indices]
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        f1 = cross_val_score(clf, X_selected, y_resampled, cv=skf, scoring='f1').mean()
        recall = cross_val_score(clf, X_selected, y_resampled, cv=skf, scoring='recall').mean()
        return f1, recall
    except:
        return 0.0, 0.0

def mate_combined(ind1, ind2):
    """Custom crossover for combined feature selection and hyperparameters"""
    # Crossover feature selection part
    for i in range(NUM_FEATURES):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    
    # Crossover hyperparameters
    if random.random() < 0.5:
        ind1[NUM_FEATURES], ind2[NUM_FEATURES] = ind2[NUM_FEATURES], ind1[NUM_FEATURES]  # k
    
    if random.random() < 0.5:
        ind1[NUM_FEATURES + 1], ind2[NUM_FEATURES + 1] = ind2[NUM_FEATURES + 1], ind1[NUM_FEATURES + 1]  # weights
    
    if random.random() < 0.5:
        ind1[NUM_FEATURES + 2], ind2[NUM_FEATURES + 2] = ind2[NUM_FEATURES + 2], ind1[NUM_FEATURES + 2]  # metric
    
    return ind1, ind2

def mutate_combined(individual, indpb=0.05):
    """Custom mutation for combined feature selection and hyperparameters"""
    # Mutate feature selection part
    for i in range(NUM_FEATURES):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]
    
    # Mutate hyperparameters
    if random.random() < 0.3:
        individual[NUM_FEATURES] = random.randint(K_RANGE[0], K_RANGE[1] - 1)
    
    if random.random() < 0.3:
        individual[NUM_FEATURES + 1] = random.choice(WEIGHTS_OPTIONS)
    
    if random.random() < 0.3:
        individual[NUM_FEATURES + 2] = random.choice(METRIC_OPTIONS)
    
    return individual,

toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", mate_combined)
toolbox.register("mutate", mutate_combined, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga(n_gen=25, mu=40, lambda_=40, elite_size=1):
    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(elite_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    for gen in range(n_gen):
        offspring = algorithms.varOr(pop, toolbox, lambda_, cxpb=0.5, mutpb=0.3)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, mu)
        hof.update(pop)
        record = stats.compile(pop)
        print(f"Gen {gen+1}: F1 avg={record['avg'][0]:.4f}, Recall avg={record['avg'][1]:.4f}, Max F1={record['max'][0]:.4f}, Max Recall={record['max'][1]:.4f}")

    return pop, stats, hof

# --- 4. Run GA for Both Feature Selection and Hyperparameter Tuning ---
N_RUNS = 10
results = []

print(f"\n🧬 Running Genetic Algorithm for Feature Selection + Hyperparameter Tuning ({N_RUNS} runs)")
print(f"📊 Original features: {NUM_FEATURES}")

for run_idx in range(N_RUNS):
    print(f"\n🔁 Run {run_idx+1}/{N_RUNS}")
    population, logbook, best = run_ga()
    best_ind = best[0]
    
    # Extract results
    feature_mask = best_ind[:NUM_FEATURES]
    best_k, best_weights, best_metric = best_ind[NUM_FEATURES:]
    best_features = [i for i in range(NUM_FEATURES) if feature_mask[i] == 1]
    selected_names = [X.columns[i] for i in best_features]

    if len(best_features) > 0:
        X_selected = X_resampled[:, best_features]
        final_clf = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, metric=best_metric)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_f1 = cross_val_score(final_clf, X_selected, y_resampled, cv=skf, scoring='f1')
        scores_recall = cross_val_score(final_clf, X_selected, y_resampled, cv=skf, scoring='recall')

        print(f"✅ Selected Features ({len(best_features)}): {selected_names}")
        print(f"✅ Best Hyperparameters:")
        print(f"   - n_neighbors: {best_k}")
        print(f"   - weights: {best_weights}")
        print(f"   - metric: {best_metric}")
        print(f"🔹 F1-score CV mean: {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")
        print(f"🔹 Recall CV mean: {scores_recall.mean():.4f} ± {scores_recall.std():.4f}")

        results.append({
            "run": run_idx + 1,
            "features_count": len(best_features),
            "selected_features": selected_names,
            "n_neighbors": best_k,
            "weights": best_weights,
            "metric": best_metric,
            "cv_f1_score": scores_f1.mean(),
            "cv_f1_std": scores_f1.std(),
            "cv_recall": scores_recall.mean(),
            "cv_recall_std": scores_recall.std()
        })
    else:
        print("❌ No features selected.")
        results.append({
            "run": run_idx + 1,
            "features_count": 0,
            "selected_features": [],
            "n_neighbors": best_k,
            "weights": best_weights,
            "metric": best_metric,
            "cv_f1_score": 0.0,
            "cv_f1_std": 0.0,
            "cv_recall": 0.0,
            "cv_recall_std": 0.0
        })

# --- 5. Analyze Combined Results ---
df_results = pd.DataFrame(results)
print("\n📊 Summary of Combined Optimization Runs:")
print(df_results[['run', 'features_count', 'n_neighbors', 'weights', 'metric', 'cv_f1_score', 'cv_f1_std', 'cv_recall', 'cv_recall_std']])

# Feature frequency
feature_frequency = {}
for _, row in df_results.iterrows():
    for feature in row['selected_features']:
        feature_frequency[feature] = feature_frequency.get(feature, 0) + 1

if feature_frequency:
    print("\n📌 Feature selection frequency:")
    for feature, count in sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {count}/{N_RUNS} ({count / N_RUNS * 100:.1f}%)")

# Hyperparameter frequency
print("\n📌 Hyperparameter frequency:")
print(f"n_neighbors distribution:")
for k in sorted(df_results['n_neighbors'].unique()):
    count = (df_results['n_neighbors'] == k).sum()
    print(f"  k={k}: {count}/{N_RUNS} ({count / N_RUNS * 100:.1f}%)")

print(f"\nweights distribution:")
for weight in df_results['weights'].unique():
    count = (df_results['weights'] == weight).sum()
    print(f"  {weight}: {count}/{N_RUNS} ({count / N_RUNS * 100:.1f}%)")

print(f"\nmetric distribution:")
for metric in df_results['metric'].unique():
    count = (df_results['metric'] == metric).sum()
    print(f"  {metric}: {count}/{N_RUNS} ({count / N_RUNS * 100:.1f}%)")

# Best run
best_run_idx = df_results['cv_f1_score'].idxmax()
best_run = df_results.iloc[best_run_idx]
print(f"\n🏆 Best Combined Optimization Run #{best_run['run']}:")
print(f" - Selected Features ({best_run['features_count']}): {best_run['selected_features']}")
print(f" - n_neighbors: {best_run['n_neighbors']}")
print(f" - weights: {best_run['weights']}")
print(f" - metric: {best_run['metric']}")
print(f" - F1-score = {best_run['cv_f1_score']:.4f} ± {best_run['cv_f1_std']:.4f}")
print(f" - Recall = {best_run['cv_recall']:.4f} ± {best_run['cv_recall_std']:.4f}")

# Global stats
print("\n📈 Overall Combined Optimization Statistics:")
print(f" - Avg F1-score: {df_results['cv_f1_score'].mean():.4f} ± {df_results['cv_f1_score'].std():.4f}")
print(f" - Avg Recall: {df_results['cv_recall'].mean():.4f} ± {df_results['cv_recall_std']:.4f}")
print(f" - Avg Features Selected: {df_results['features_count'].mean():.1f}")
print(f" - Feature Reduction: {(1 - df_results['features_count'].mean() / NUM_FEATURES) * 100:.1f}%")
print(f" - Most common n_neighbors: {df_results['n_neighbors'].mode().iloc[0]}")
print(f" - Most common weights: {df_results['weights'].mode().iloc[0]}")
print(f" - Most common metric: {df_results['metric'].mode().iloc[0]}")

# --- 6. Save Results and Models ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create Excel file with multiple sheets for metrics
excel_filename = f"knn_ga_combined_metrics_{timestamp}.xlsx"

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Sheet 1: Metrics Summary
    metrics_summary = pd.DataFrame({
        'Run': df_results['run'],
        'Features_Count': df_results['features_count'],
        'N_Neighbors': df_results['n_neighbors'],
        'Weights': df_results['weights'],
        'Metric': df_results['metric'],
        'F1_Score_Mean': df_results['cv_f1_score'],
        'F1_Score_Std': df_results['cv_f1_std'],
        'Recall_Mean': df_results['cv_recall'],
        'Recall_Std': df_results['cv_recall_std']
    })
    metrics_summary.to_excel(writer, sheet_name='Metrics_Summary', index=False)
    
    # Sheet 2: Detailed Results
    df_results.to_excel(writer, sheet_name='Detailed_Results', index=False)
    
    # Sheet 3: Summary Statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Average F1-Score', 'F1-Score Std', 'Average Recall', 'Recall Std', 
                   'Avg Features Selected', 'Feature Reduction %', 'Best F1-Score', 'Best Recall',
                   'Most Common K', 'Most Common Weights', 'Most Common Metric'],
        'Value': [
            df_results['cv_f1_score'].mean(),
            df_results['cv_f1_score'].std(),
            df_results['cv_recall'].mean(),
            df_results['cv_recall'].std(),
            df_results['features_count'].mean(),
            (1 - df_results['features_count'].mean() / NUM_FEATURES) * 100,
            best_run['cv_f1_score'],
            best_run['cv_recall'],
            df_results['n_neighbors'].mode().iloc[0],
            df_results['weights'].mode().iloc[0],
            df_results['metric'].mode().iloc[0]
        ]
    })
    summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)

print(f"\n💾 Excel metrics report saved to: {excel_filename}")

# Save features and parameters Excel
features_params_excel_filename = f"knn_ga_combined_features_params_{timestamp}.xlsx"

with pd.ExcelWriter(features_params_excel_filename, engine='openpyxl') as writer:
    # Sheet 1: Best Run Configuration
    best_config = pd.DataFrame({
        'Parameter': ['Run Number', 'N_Neighbors', 'Weights', 'Metric', 'Features Count', 'F1 Score', 'Recall'],
        'Value': [
            best_run['run'],
            best_run['n_neighbors'],
            best_run['weights'],
            best_run['metric'],
            best_run['features_count'],
            f"{best_run['cv_f1_score']:.4f} ± {best_run['cv_f1_std']:.4f}",
            f"{best_run['cv_recall']:.4f} ± {best_run['cv_recall_std']:.4f}"
        ]
    })
    best_config.to_excel(writer, sheet_name='Best_Configuration', index=False)
    
    # Sheet 2: Feature Analysis
    if feature_frequency:
        feature_analysis = pd.DataFrame([
            {'Feature_Name': feature, 'Selection_Count': count, 'Selection_Percentage': (count / N_RUNS * 100)}
            for feature, count in sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)
        ])
        feature_analysis.to_excel(writer, sheet_name='Feature_Analysis', index=False)
        
        # Sheet 3: All Features Status
        all_features_df = pd.DataFrame({
            'Feature_Index': list(range(NUM_FEATURES)),
            'Feature_Name': X.columns.tolist(),
            'Selection_Count': [feature_frequency.get(feature, 0) for feature in X.columns],
            'Selection_Percentage': [(feature_frequency.get(feature, 0) / N_RUNS * 100) for feature in X.columns],
            'Selected_in_Best_Run': ['Yes' if feature in (best_run['selected_features'] or []) else 'No' for feature in X.columns]
        })
        all_features_df = all_features_df.sort_values('Selection_Count', ascending=False)
        all_features_df.to_excel(writer, sheet_name='All_Features_Status', index=False)
    
    # Sheet 4: Hyperparameter Analysis
    hyperparams_analysis = []
    
    # K analysis
    for k in sorted(df_results['n_neighbors'].unique()):
        count = (df_results['n_neighbors'] == k).sum()
        hyperparams_analysis.append({
            'Parameter': 'n_neighbors',
            'Value': k,
            'Count': count,
            'Percentage': (count / N_RUNS * 100)
        })
    
    # Weights analysis
    for weight in df_results['weights'].unique():
        count = (df_results['weights'] == weight).sum()
        hyperparams_analysis.append({
            'Parameter': 'weights',
            'Value': weight,
            'Count': count,
            'Percentage': (count / N_RUNS * 100)
        })
    
    # Metric analysis
    for metric in df_results['metric'].unique():
        count = (df_results['metric'] == metric).sum()
        hyperparams_analysis.append({
            'Parameter': 'metric',
            'Value': metric,
            'Count': count,
            'Percentage': (count / N_RUNS * 100)
        })
    
    hyperparams_df = pd.DataFrame(hyperparams_analysis)
    hyperparams_df.to_excel(writer, sheet_name='Hyperparameters_Analysis', index=False)

print(f"💾 Features and parameters Excel saved to: {features_params_excel_filename}")

# Train and save the best model
print(f"\n🎯 Training and saving the best model...")
if best_run['selected_features']:
    best_feature_indices = [X.columns.get_loc(feature) for feature in best_run['selected_features']]
    X_best_selected = X_resampled[:, best_feature_indices]
    
    best_model = KNeighborsClassifier(
        n_neighbors=best_run['n_neighbors'],
        weights=best_run['weights'],
        metric=best_run['metric']
    )
    best_model.fit(X_best_selected, y_resampled)
    
    # Save the best model
    model_filename = f"knn_ga_combined_model_{timestamp}.pkl"
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'selected_features': best_run['selected_features'],
        'selected_feature_indices': best_feature_indices,
        'feature_names': X.columns.tolist(),
        'hyperparameters': {
            'n_neighbors': best_run['n_neighbors'],
            'weights': best_run['weights'],
            'metric': best_run['metric']
        },
        'performance': {
            'cv_f1_score': best_run['cv_f1_score'],
            'cv_f1_std': best_run['cv_f1_std'],
            'cv_recall': best_run['cv_recall'],
            'cv_recall_std': best_run['cv_recall_std']
        },
        'experiment_info': {
            'type': 'Combined_Feature_Selection_and_Hyperparameter_Tuning',
            'timestamp': timestamp,
            'run_number': best_run['run'],
            'total_runs': N_RUNS,
            'original_features': NUM_FEATURES,
            'selected_features_count': len(best_run['selected_features'])
        }
    }
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"💾 Best model saved to: {model_filename}")
else:
    print("❌ Cannot save model: No features were selected in the best run")
    model_filename = None

# Download files (for Google Colab)
print(f"\n📥 Downloading files...")
files.download(excel_filename)
files.download(features_params_excel_filename)
if model_filename:
    files.download(model_filename)

print(f"\n✅ All files saved and downloaded successfully!")
print(f"📊 Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 7. Model Usage Example ---
if model_filename:
    print(f"\n📖 Model Usage Example:")
    print(f"```python")
    print(f"import pickle")
    print(f"import numpy as np")
    print(f"")
    print(f"# Load the best model")
    print(f"with open('{model_filename}', 'rb') as f:")
    print(f"    model_data = pickle.load(f)")
    print(f"")
    print(f"model = model_data['model']")
    print(f"scaler = model_data['scaler']")
    print(f"selected_features = model_data['selected_features']")
    print(f"selected_indices = model_data['selected_feature_indices']")
    print(f"hyperparameters = model_data['hyperparameters']")
    print(f"")
    print(f"# Use the model for prediction")
    print(f"# X_new_scaled = scaler.transform(X_new)")
    print(f"# X_new_selected = X_new_scaled[:, selected_indices]")
    print(f"# predictions = model.predict(X_new_selected)")
    print(f"```")
