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

# --- 3. GA Setup for Hyperparameter Tuning ---
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
    """Create an individual representing KNN hyperparameters"""
    k = random.randint(K_RANGE[0], K_RANGE[1] - 1)
    weights = random.choice(WEIGHTS_OPTIONS)
    metric = random.choice(METRIC_OPTIONS)
    return [k, weights, metric]

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_fitness(individual):
    k, weights, metric = individual
    
    # Use all features for hyperparameter tuning
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        f1 = cross_val_score(clf, X_resampled, y_resampled, cv=skf, scoring='f1').mean()
        recall = cross_val_score(clf, X_resampled, y_resampled, cv=skf, scoring='recall').mean()
        return f1, recall
    except:
        return 0.0, 0.0

def mate_hyperparams(ind1, ind2):
    """Custom crossover for hyperparameters"""
    # Crossover k value
    if random.random() < 0.5:
        ind1[0], ind2[0] = ind2[0], ind1[0]
    
    # Crossover weights
    if random.random() < 0.5:
        ind1[1], ind2[1] = ind2[1], ind1[1]
    
    # Crossover metric
    if random.random() < 0.5:
        ind1[2], ind2[2] = ind2[2], ind1[2]
    
    return ind1, ind2

def mutate_hyperparams(individual, indpb=0.3):
    """Custom mutation for hyperparameters"""
    if random.random() < indpb:
        individual[0] = random.randint(K_RANGE[0], K_RANGE[1] - 1)
    
    if random.random() < indpb:
        individual[1] = random.choice(WEIGHTS_OPTIONS)
    
    if random.random() < indpb:
        individual[2] = random.choice(METRIC_OPTIONS)
    
    return individual,

toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", mate_hyperparams)
toolbox.register("mutate", mutate_hyperparams, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga(n_gen=20, mu=30, lambda_=30, elite_size=1):
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

# --- 4. Run GA for Hyperparameter Tuning ---
N_RUNS = 10
results = []

print(f"\n🧬 Running Genetic Algorithm for Hyperparameter Tuning ({N_RUNS} runs)")
print(f"📊 Using all {X_resampled.shape[1]} features")

for run_idx in range(N_RUNS):
    print(f"\n🔁 Run {run_idx+1}/{N_RUNS}")
    population, logbook, best = run_ga()
    best_ind = best[0]
    best_k, best_weights, best_metric = best_ind

    # Evaluate best hyperparameters
    final_clf = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, metric=best_metric)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_f1 = cross_val_score(final_clf, X_resampled, y_resampled, cv=skf, scoring='f1')
    scores_recall = cross_val_score(final_clf, X_resampled, y_resampled, cv=skf, scoring='recall')

    print(f"✅ Best Hyperparameters:")
    print(f"   - n_neighbors: {best_k}")
    print(f"   - weights: {best_weights}")
    print(f"   - metric: {best_metric}")
    print(f"🔹 F1-score CV mean: {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")
    print(f"🔹 Recall CV mean: {scores_recall.mean():.4f} ± {scores_recall.std():.4f}")

    results.append({
        "run": run_idx + 1,
        "n_neighbors": best_k,
        "weights": best_weights,
        "metric": best_metric,
        "cv_f1_score": scores_f1.mean(),
        "cv_f1_std": scores_f1.std(),
        "cv_recall": scores_recall.mean(),
        "cv_recall_std": scores_recall.std()
    })

# --- 5. Analyze Hyperparameter Tuning Results ---
df_results = pd.DataFrame(results)
print("\n📊 Summary of Hyperparameter Tuning Runs:")
print(df_results[['run', 'n_neighbors', 'weights', 'metric', 'cv_f1_score', 'cv_f1_std', 'cv_recall', 'cv_recall_std']])

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
print(f"\n🏆 Best Hyperparameter Tuning Run #{best_run['run']}:")
print(f" - n_neighbors: {best_run['n_neighbors']}")
print(f" - weights: {best_run['weights']}")
print(f" - metric: {best_run['metric']}")
print(f" - F1-score = {best_run['cv_f1_score']:.4f} ± {best_run['cv_f1_std']:.4f}")
print(f" - Recall = {best_run['cv_recall']:.4f} ± {best_run['cv_recall_std']:.4f}")

# Global stats
print("\n📈 Overall Hyperparameter Tuning Statistics:")
print(f" - Avg F1-score: {df_results['cv_f1_score'].mean():.4f} ± {df_results['cv_f1_score'].std():.4f}")
print(f" - Avg Recall: {df_results['cv_recall'].mean():.4f} ± {df_results['cv_recall_std']:.4f}")
print(f" - Most common n_neighbors: {df_results['n_neighbors'].mode().iloc[0]}")
print(f" - Most common weights: {df_results['weights'].mode().iloc[0]}")
print(f" - Most common metric: {df_results['metric'].mode().iloc[0]}")

# --- 5. Train final model with best hyperparameters ---
print(f"\n🎯 Training final model with best hyperparameters...")
final_clf = KNeighborsClassifier(
    n_neighbors=int(best_run['n_neighbors']),
    weights=best_run['weights'],
    metric=best_run['metric']
)
final_clf.fit(X_resampled, y_resampled)

print(f"✅ Final model trained with:")
print(f"   • n_neighbors: {best_run['n_neighbors']}")
print(f"   • weights: {best_run['weights']}")
print(f"   • metric: {best_run['metric']}")

# --- 6. Generate Excel Reports ---
print("\n📊 Generating Excel reports...")

# Create timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Metrics Report ---
excel_filename = f'knn_ga_hyperparameter_metrics_{timestamp}.xlsx'

# Summary statistics
summary_stats = pd.DataFrame({
    'Metric': ['F1_Score', 'Recall'],
    'Best_Mean': [best_run['cv_f1_score'], best_run['cv_recall']],
    'Best_Std': [best_run['cv_f1_std'], best_run['cv_recall_std']],
    'Overall_Mean': [df_results['cv_f1_score'].mean(), df_results['cv_recall'].mean()],
    'Overall_Std': [df_results['cv_f1_score'].std(), df_results['cv_recall'].std()],
    'Overall_Min': [df_results['cv_f1_score'].min(), df_results['cv_recall'].min()],
    'Overall_Max': [df_results['cv_f1_score'].max(), df_results['cv_recall'].max()]
})

# Create Excel file with multiple sheets
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Summary sheet
    summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    # All runs detailed results
    df_results.to_excel(writer, sheet_name='All_Runs_Details', index=False)
    
    # Best run info
    best_run_info = pd.DataFrame({
        'Parameter': ['Run_Number', 'n_neighbors', 'weights', 'metric', 'F1_Mean', 'F1_Std', 'Recall_Mean', 'Recall_Std'],
        'Value': [best_run['run'], best_run['n_neighbors'], best_run['weights'], best_run['metric'],
                 best_run['cv_f1_score'], best_run['cv_f1_std'], best_run['cv_recall'], best_run['cv_recall_std']]
    })
    best_run_info.to_excel(writer, sheet_name='Best_Run_Info', index=False)

print(f"✅ Excel report saved: {excel_filename}")

# Download the Excel file in Colab
files.download(excel_filename)

# --- Hyperparameters Analysis Report ---
hyperparams_excel_filename = f'knn_ga_hyperparameter_analysis_{timestamp}.xlsx'

# Hyperparameter distribution analysis
n_neighbors_dist = df_results['n_neighbors'].value_counts().reset_index()
n_neighbors_dist.columns = ['n_neighbors', 'frequency']
n_neighbors_dist = n_neighbors_dist.sort_values('n_neighbors')

weights_dist = df_results['weights'].value_counts().reset_index()
weights_dist.columns = ['weights', 'frequency']

metric_dist = df_results['metric'].value_counts().reset_index()
metric_dist.columns = ['metric', 'frequency']

# Performance by hyperparameter value
n_neighbors_perf = df_results.groupby('n_neighbors').agg({
    'cv_f1_score': ['mean', 'std', 'min', 'max', 'count'],
    'cv_recall': ['mean', 'std', 'min', 'max']
}).round(4)
n_neighbors_perf.columns = ['F1_Mean', 'F1_Std', 'F1_Min', 'F1_Max', 'Count', 'Recall_Mean', 'Recall_Std', 'Recall_Min', 'Recall_Max']
n_neighbors_perf = n_neighbors_perf.reset_index()

weights_perf = df_results.groupby('weights').agg({
    'cv_f1_score': ['mean', 'std', 'min', 'max', 'count'],
    'cv_recall': ['mean', 'std', 'min', 'max']
}).round(4)
weights_perf.columns = ['F1_Mean', 'F1_Std', 'F1_Min', 'F1_Max', 'Count', 'Recall_Mean', 'Recall_Std', 'Recall_Min', 'Recall_Max']
weights_perf = weights_perf.reset_index()

metric_perf = df_results.groupby('metric').agg({
    'cv_f1_score': ['mean', 'std', 'min', 'max', 'count'],
    'cv_recall': ['mean', 'std', 'min', 'max']
}).round(4)
metric_perf.columns = ['F1_Mean', 'F1_Std', 'F1_Min', 'F1_Max', 'Count', 'Recall_Mean', 'Recall_Std', 'Recall_Min', 'Recall_Max']
metric_perf = metric_perf.reset_index()

# Create hyperparameters analysis Excel file
with pd.ExcelWriter(hyperparams_excel_filename, engine='openpyxl') as writer:
    # Distributions
    n_neighbors_dist.to_excel(writer, sheet_name='n_neighbors_Distribution', index=False)
    weights_dist.to_excel(writer, sheet_name='weights_Distribution', index=False)
    metric_dist.to_excel(writer, sheet_name='metric_Distribution', index=False)
    
    # Performance by hyperparameter
    n_neighbors_perf.to_excel(writer, sheet_name='n_neighbors_Performance', index=False)
    weights_perf.to_excel(writer, sheet_name='weights_Performance', index=False)
    metric_perf.to_excel(writer, sheet_name='metric_Performance', index=False)
      # GA settings
    ga_settings = pd.DataFrame({
        'Setting': ['Population_Size', 'Generations', 'Crossover_Probability', 'Mutation_Probability', 'Total_Runs'],
        'Value': [30, 20, 0.5, 0.3, N_RUNS]  # Default values from run_ga function
    })
    ga_settings.to_excel(writer, sheet_name='GA_Settings', index=False)

print(f"✅ Hyperparameters analysis saved: {hyperparams_excel_filename}")

# Download the hyperparameters analysis file in Colab
files.download(hyperparams_excel_filename)

# --- 7. Save Model ---
print("\n💾 Saving model...")

model_filename = f'knn_ga_hyperparameter_model_{timestamp}.pkl'

# Package everything needed for prediction
model_package = {
    'model': final_clf,
    'scaler': scaler,
    'smote': sm,
    'feature_names': list(X.columns),
    'target_column': target_col,
    'best_hyperparameters': {
        'n_neighbors': int(best_run['n_neighbors']),
        'weights': best_run['weights'],
        'metric': best_run['metric']
    },
    'performance_metrics': {
        'best_f1_mean': best_run['cv_f1_score'],
        'best_f1_std': best_run['cv_f1_std'],
        'best_recall_mean': best_run['cv_recall'],
        'best_recall_std': best_run['cv_recall_std'],
        'overall_f1_mean': df_results['cv_f1_score'].mean(),
        'overall_f1_std': df_results['cv_f1_score'].std(),
        'overall_recall_mean': df_results['cv_recall'].mean(),
        'overall_recall_std': df_results['cv_recall'].std()
    },    'ga_settings': {
        'population_size': 30,  # Default mu value from run_ga function
        'generations': 20,      # Default n_gen value from run_ga function
        'crossover_prob': 0.5,  # Default cxpb value from run_ga function
        'mutation_prob': 0.3,   # Default mutpb value from run_ga function
        'total_runs': N_RUNS
    },
    'all_runs_results': df_results.to_dict('records'),
    'dataset_info': {
        'original_shape': X.shape,
        'after_smote_shape': X_resampled.shape,
        'target_distribution': dict(zip(*np.unique(y_resampled, return_counts=True)))
    },
    'timestamp': timestamp
}

# Save the model package
with open(model_filename, 'wb') as f:
    pickle.dump(model_package, f)

print(f"✅ Model saved: {model_filename}")

# Download the model file in Colab
files.download(model_filename)

print("\n🎯 Process completed successfully!")
print(f"📁 Files generated:")
print(f"   • {excel_filename} - Performance metrics report")
print(f"   • {hyperparams_excel_filename} - Hyperparameters analysis report")
print(f"   • {model_filename} - Trained model package")

# --- Usage Example ---
print("\n📖 Usage example for loading the saved model:")
print(f"""
# Load the saved model
import pickle
with open('{model_filename}', 'rb') as f:
    model_package = pickle.load(f)

# Extract components
clf = model_package['model']
scaler = model_package['scaler']
best_hyperparams = model_package['best_hyperparameters']
feature_names = model_package['feature_names']

# Check best hyperparameters
print("Best hyperparameters found:")
print(f"n_neighbors: {{best_hyperparams['n_neighbors']}}")
print(f"weights: {{best_hyperparams['weights']}}")
print(f"metric: {{best_hyperparams['metric']}}")

# For new predictions:
# 1. Load your new data
# 2. Scale it: X_new_scaled = scaler.transform(X_new)
# 3. Predict: predictions = clf.predict(X_new_scaled)
""")

