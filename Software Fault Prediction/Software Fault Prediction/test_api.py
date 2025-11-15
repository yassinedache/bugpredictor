"""
Test script for the Software Defect Prediction API.
This script demonstrates how to use the API endpoints.
"""

import requests
import json
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running")
            print(f"Health check response: {response.json()}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

def create_sample_model():
    """Create a sample trained model for testing."""
    print("🔨 Creating sample Random Forest model...")
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Train a simple Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    model_path = "sample_rf_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Sample model saved to {model_path}")
    return model_path

def create_sample_dataset():
    """Create a sample dataset for testing."""
    print("📊 Creating sample dataset...")
    
    # Generate sample data
    np.random.seed(42)
    data = {
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(1, 2, 100),
        'feature_3': np.random.exponential(1, 100),
        'feature_4': np.random.uniform(-1, 1, 100),
        'defect': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    dataset_path = "sample_dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    print(f"✅ Sample dataset saved to {dataset_path}")
    return dataset_path

def test_model_upload():
    """Test model upload endpoint."""
    print("\n📤 Testing model upload...")
    
    model_path = create_sample_model()
    
    with open(model_path, 'rb') as f:
        files = {'file': ('sample_rf_model.pkl', f, 'application/octet-stream')}
        response = requests.post(f"{BASE_URL}/models/rf", files=files)
    
    if response.status_code == 200:
        print("✅ Model upload successful")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Model upload failed: {response.status_code}")
        print(f"Error: {response.text}")
    
    # Clean up
    os.remove(model_path)
    return response.status_code == 200

def test_dataset_upload():
    """Test dataset upload endpoint."""
    print("\n📤 Testing dataset upload...")
    
    dataset_path = create_sample_dataset()
    
    with open(dataset_path, 'rb') as f:
        files = {'file': ('sample_dataset.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/datasets", files=files)
    
    if response.status_code == 200:
        print("✅ Dataset upload successful")
        data = response.json()
        print(f"Dataset ID: {data['dataset_id']}")
        print(f"Rows: {data['rows']}, Columns: {data['columns']}")
        
        # Clean up
        os.remove(dataset_path)
        return data['dataset_id']
    else:
        print(f"❌ Dataset upload failed: {response.status_code}")
        print(f"Error: {response.text}")
        os.remove(dataset_path)
        return None

def test_prediction(dataset_id):
    """Test prediction endpoint."""
    print("\n🔮 Testing prediction...")
    
    payload = {
        'model_name': 'rf',
        'dataset_id': dataset_id,
        'threshold': 0.5
    }
    
    response = requests.post(f"{BASE_URL}/predict", data=payload)
    
    if response.status_code == 200:
        print("✅ Prediction successful")
        data = response.json()
        print(f"Accuracy: {data['metrics']['accuracy']:.4f}")
        print(f"Precision: {data['metrics']['precision']:.4f}")
        print(f"Recall: {data['metrics']['recall']:.4f}")
        print(f"F1-score: {data['metrics']['f1_score']:.4f}")
        print(f"Total predictions: {data['total_predictions']}")
        return True
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_list_endpoints():
    """Test list models and datasets endpoints."""
    print("\n📋 Testing list endpoints...")
    
    # Test list models
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Models listed: {data['total']} models found")
    else:
        print(f"❌ List models failed: {response.status_code}")
    
    # Test list datasets
    response = requests.get(f"{BASE_URL}/datasets")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Datasets listed: {data['total']} datasets found")
    else:
        print(f"❌ List datasets failed: {response.status_code}")

def run_all_tests():
    """Run all API tests."""
    print("🧪 Starting API tests...")
    print("=" * 60)
    
    # Test API health
    if not test_api_health():
        print("\n❌ API is not accessible. Please start the server first.")
        print("Run: python start_server.py")
        return False
    
    # Test model upload
    if not test_model_upload():
        print("\n❌ Model upload test failed")
        return False
    
    # Test dataset upload
    dataset_id = test_dataset_upload()
    if not dataset_id:
        print("\n❌ Dataset upload test failed")
        return False
    
    # Test prediction
    if not test_prediction(dataset_id):
        print("\n❌ Prediction test failed")
        return False
    
    # Test list endpoints
    test_list_endpoints()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("📖 Visit http://localhost:8000/docs for interactive API documentation")
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)
