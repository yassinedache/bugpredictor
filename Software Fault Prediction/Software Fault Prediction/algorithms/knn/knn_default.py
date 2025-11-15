# --- Install requirements ---
!pip install imblearn scipy openpyxl

# --- Imports ---
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

print(f"📊 Dataset shape after SMOTE: {X_resampled.shape}")
print(f"📊 Target distribution: {np.bincount(y_resampled)}")

# --- 3. Train KNN with default parameters ---
print("\n🔍 Training KNN with default parameters...")
clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
scores_f1 = cross_val_score(clf, X_resampled, y_resampled, cv=skf, scoring='f1')
scores_recall = cross_val_score(clf, X_resampled, y_resampled, cv=skf, scoring='recall')
scores_accuracy = cross_val_score(clf, X_resampled, y_resampled, cv=skf, scoring='accuracy')

print(f"\n📈 Cross-Validation Results:")
print(f"🔹 F1-score: {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")
print(f"🔹 Recall: {scores_recall.mean():.4f} ± {scores_recall.std():.4f}")
print(f"🔹 Accuracy: {scores_accuracy.mean():.4f} ± {scores_accuracy.std():.4f}")

# Train final model
print("\n🎯 Training final model on full dataset...")
clf.fit(X_resampled, y_resampled)

print(f"\n✅ Model trained successfully!")
print(f"📊 Using all {X_resampled.shape[1]} features")
print(f"📊 Default KNN parameters: n_neighbors=5, weights='uniform', metric='minkowski'")

# --- 4. Generate Excel Reports ---
print("\n📊 Generating Excel reports...")

# Create timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Metrics Report ---
excel_filename = f'knn_default_metrics_{timestamp}.xlsx'

# Detailed results for each fold
detailed_results = []
for i, (f1, recall, acc) in enumerate(zip(scores_f1, scores_recall, scores_accuracy), 1):
    detailed_results.append({
        'Fold': i,
        'F1_Score': f1,
        'Recall': recall,
        'Accuracy': acc
    })

detailed_df = pd.DataFrame(detailed_results)

# Summary statistics
summary_stats = pd.DataFrame({
    'Metric': ['F1_Score', 'Recall', 'Accuracy'],
    'Mean': [scores_f1.mean(), scores_recall.mean(), scores_accuracy.mean()],
    'Std': [scores_f1.std(), scores_recall.std(), scores_accuracy.std()],
    'Min': [scores_f1.min(), scores_recall.min(), scores_accuracy.min()],
    'Max': [scores_f1.max(), scores_recall.max(), scores_accuracy.max()]
})

# Create Excel file with multiple sheets
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Summary sheet
    summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    # Detailed results sheet
    detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
    
    # Model info sheet
    model_info = pd.DataFrame({
        'Parameter': ['Algorithm', 'n_neighbors', 'weights', 'metric', 'Total_Features', 
                     'Dataset_Shape_Original', 'Dataset_Shape_After_SMOTE'],
        'Value': ['KNN (Default)', 5, 'uniform', 'minkowski', X_resampled.shape[1],
                 f"{X.shape[0]}x{X.shape[1]}", f"{X_resampled.shape[0]}x{X_resampled.shape[1]}"]
    })
    model_info.to_excel(writer, sheet_name='Model_Info', index=False)

print(f"✅ Excel report saved: {excel_filename}")

# Download the Excel file in Colab
files.download(excel_filename)

# --- 5. Save Model ---
print("\n💾 Saving model...")

model_filename = f'knn_default_model_{timestamp}.pkl'

# Package everything needed for prediction
model_package = {
    'model': clf,
    'scaler': scaler,
    'smote': sm,
    'feature_names': list(X.columns),
    'target_column': target_col,
    'model_params': {
        'n_neighbors': 5,
        'weights': 'uniform',
        'metric': 'minkowski'
    },
    'performance_metrics': {
        'f1_mean': scores_f1.mean(),
        'f1_std': scores_f1.std(),
        'recall_mean': scores_recall.mean(),
        'recall_std': scores_recall.std(),
        'accuracy_mean': scores_accuracy.mean(),
        'accuracy_std': scores_accuracy.std()
    },
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
feature_names = model_package['feature_names']

# For new predictions:
# 1. Load your new data
# 2. Scale it: X_new_scaled = scaler.transform(X_new)
# 3. Predict: predictions = clf.predict(X_new_scaled)
""")
