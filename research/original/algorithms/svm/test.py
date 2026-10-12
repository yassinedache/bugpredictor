import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE
from deap import base, creator, tools, algorithms
from scipy.io import arff
from google.colab import files
from io import StringIO


# --- 1. Charger le fichier .arff via upload ---
print("Uploader le fichier .arff")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

content = uploaded[filename].decode('utf-8')  # Décoder bytes en str
data, meta = arff.loadarff(StringIO(content))
df = pd.DataFrame(data)

# --- 2. Diagnostic ---
print("Colonnes du dataset:")
print(df.columns)

print("\nRésumé des colonnes avec types et valeurs uniques:")
for col in df.columns:
    print(f"\nColonne '{col}' ({df[col].dtype}):")
    print(df[col].value_counts(dropna=False))

print("\nAffichage des 10 premières lignes:")
print(df.head(10))

# --- 3. Définir la colonne cible ---
target_col = 'bug'  # <-- adapter selon ton dataset

if df[target_col].dtype == object or df[target_col].dtype.name == 'bytes':
    df[target_col] = df[target_col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    df[target_col] = df[target_col].apply(lambda x: 1 if x in ['Y', 'yes', 'true', 'True', 1] else 0)

if df[target_col].nunique() < 2:
    raise ValueError(f"❌ La variable cible '{target_col}' contient une seule classe. Impossible de poursuivre.")

# --- 4. Préparation des données ---
X = df.drop(columns=[target_col])
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_scaled, y = sm.fit_resample(X_scaled, y)

# --- 5. Diviser en ensembles d'entraînement et de test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- 6. Configuration GA ---
NUM_FEATURES = X_train.shape[1]

# Hyperparamètres C et gamma (log scale exponents)
C_MIN, C_MAX = 0, 1     # log10(C) entre 10^-3 et 10^3
GAMMA_MIN, GAMMA_MAX = -1, 0  # log10(gamma) entre 10^-4 et 10^1

CHROMOSOME_LENGTH = NUM_FEATURES + 5 + 5 + 2  # +2 bits for kernel

# Réinitialiser creator si déjà défini
if 'FitnessMulti' in creator.__dict__:
    del creator.FitnessMulti
if 'Individual' in creator.__dict__:
    del creator.Individual

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # maximiser F1 et Recall
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=CHROMOSOME_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_hyperparameters(individual):
    c_bits = individual[NUM_FEATURES:NUM_FEATURES+5]
    gamma_bits = individual[NUM_FEATURES+5:NUM_FEATURES+10]
    kernel_bits = individual[NUM_FEATURES+10:NUM_FEATURES+12]

    c_int = int("".join(str(bit) for bit in c_bits), 2)
    gamma_int = int("".join(str(bit) for bit in gamma_bits), 2)
    kernel_int = int("".join(str(bit) for bit in kernel_bits), 2)

    c_exp = C_MIN + (c_int / 31) * (C_MAX - C_MIN)
    gamma_exp = GAMMA_MIN + (gamma_int / 31) * (GAMMA_MAX - GAMMA_MIN)

    C_val = 10 ** c_exp
    gamma_val = 10 ** gamma_exp

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel = kernels[kernel_int % 4]  # Ensure it's always a valid index

    return C_val, gamma_val, kernel

def eval_fitness(individual):
    selected_indices = [i for i in range(NUM_FEATURES) if individual[i] == 1]
    if len(selected_indices) == 0:
        return 0.0, 0.0

    X_selected = X_train[:, selected_indices]  # Utiliser X_train au lieu de X_scaled
    C_val, gamma_val, kernel = decode_hyperparameters(individual)

    clf = SVC(
        C=C_val,
        gamma=gamma_val,
        kernel=kernel,
        class_weight='balanced',
        random_state=42
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(clf, X_selected, y_train, cv=skf, scoring='f1').mean()
    recall = cross_val_score(clf, X_selected, y_train, cv=skf, scoring='recall').mean()

    return f1, recall

toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga(n_gen=20, mu=30, lambda_=30, elite_size=1):
    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(elite_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Evaluate initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    for gen in range(n_gen):
        offspring = algorithms.varOr(pop, toolbox, lambda_, cxpb=0.5, mutpb=0.2)

        # Evaluate the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Combine and select the next generation (elitism)
        pop = toolbox.select(pop + offspring, mu)
        hof.update(pop)
        record = stats.compile(pop)
        print(f"Gen {gen+1}: {record}")

    return pop, stats, hof


# --- 7. Lancer le GA ---
N_RUNS = 10
results = []

for run_idx in range(N_RUNS):
    print(f"\n--- Run {run_idx+1} / {N_RUNS} ---")
    population, logbook, best = run_ga()
    best_ind = best[0]

    best_features = [i for i in range(NUM_FEATURES) if best_ind[i] == 1]
    C_opt, gamma_opt, kernel_opt = decode_hyperparameters(best_ind)
    selected_names = [X.columns[i] for i in best_features]

    if len(best_features) > 0:
        X_train_best = X_train[:, best_features]
        X_test_best = X_test[:, best_features]
        
        # Évaluation par validation croisée sur l'ensemble d'entraînement
        final_clf = SVC(
            C=C_opt,
            gamma=gamma_opt,
            kernel=kernel_opt,
            class_weight='balanced',
            random_state=42
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_f1 = cross_val_score(final_clf, X_train_best, y_train, cv=skf, scoring='f1')
        scores_recall = cross_val_score(final_clf, X_train_best, y_train, cv=skf, scoring='recall')
        
        # Entraîner le modèle final sur tout l'ensemble d'entraînement
        final_clf.fit(X_train_best, y_train)
        
        # Évaluer sur l'ensemble de test
        y_pred = final_clf.predict(X_test_best)
        test_f1 = f1_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)

        print(f"🛠️ Hyperparamètres optimaux: C={C_opt:.5f}, gamma={gamma_opt:.5f}, kernel={kernel_opt}")
        print(f"🎯 Meilleures features sélectionnées ({len(best_features)}): {selected_names}")
        print(f"✅ F1-score moyen (validation croisée): {scores_f1.mean():.4f}")
        print(f"✅ Recall moyen (validation croisée): {scores_recall.mean():.4f}")
        print(f"💯 Test set F1-score: {test_f1:.4f}")
        print(f"💯 Test set Recall: {test_recall:.4f}")

        results.append({
            "run": run_idx + 1,
            "C": C_opt,
            "gamma": gamma_opt,
            "kernel": kernel_opt,
            "features_count": len(best_features),
            "cv_f1_score": scores_f1.mean(),
            "cv_recall": scores_recall.mean(),
            "test_f1_score": test_f1,
            "test_recall": test_recall
        })
    else:
        print("❌ Aucune feature sélectionnée pour ce run.")
        results.append({
            "run": run_idx + 1,
            "C": None,
            "gamma": None,
            "kernel": None,
            "features_count": 0,
            "cv_f1_score": 0.0,
            "cv_recall": 0.0,
            "test_f1_score": 0.0,
            "test_recall": 0.0
        })

print("\n=== Résumé des 10 runs ===")
df_results = pd.DataFrame(results)
print(df_results)

# --- 8. Sélectionner et sauvegarder le meilleur modèle ---
if not df_results.empty:
    # Sélectionner le meilleur modèle basé sur la performance de test
    best_run_idx = df_results['test_f1_score'].idxmax()
    best_run = df_results.iloc[best_run_idx]
    
    print(f"\n🏆 Meilleur modèle (Run {best_run['run']}):")
    print(f"   - C = {best_run['C']:.5f}")
    print(f"   - gamma = {best_run['gamma']:.5f}")
    print(f"   - kernel = {best_run['kernel']}")
    print(f"   - Nombre de features = {best_run['features_count']}")
    print(f"   - F1-score (test) = {best_run['test_f1_score']:.4f}")
    print(f"   - Recall (test) = {best_run['test_recall']:.4f}")