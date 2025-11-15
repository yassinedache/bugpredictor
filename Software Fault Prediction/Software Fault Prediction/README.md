# Software Defect Prediction API

A complete FastAPI backend for software defect prediction with machine learning model management, dataset ingestion, and prediction capabilities.

## 🚀 Features

- **Model Management**: Upload and manage ML models (SVM, KNN, Random Forest, Logistic Regression) in pickle format
- **Dataset Ingestion**: Support for CSV and ARFF format datasets with automatic preprocessing
- **Prediction Engine**: Run predictions with comprehensive metrics and feature importance
- **CORS Support**: Ready for frontend integration
- **Auto Documentation**: Interactive API docs at `/docs`
- **Error Handling**: Comprehensive error handling and logging

## 📁 Project Structure

```
Interface/
├── app.py                 # Main FastAPI application
├── preprocessing.py       # Data preprocessing utilities
├── requirements.txt       # Python dependencies
├── start_server.py       # Server startup script
├── test_api.py           # API testing script
├── README.md             # This file
├── models/               # Directory for uploaded models
└── datasets/             # Directory for uploaded datasets
```

## 🛠️ Setup

### 1. Install Dependencies

```powershell
# Navigate to the Interface directory
cd c:\Users\dell\Desktop\ML_Project\software-defect-prediction\Interface

# Install required packages
pip install -r requirements.txt
```

### 2. Start the Server

```powershell
# Option 1: Using the startup script (recommended)
python start_server.py

# Option 2: Using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API server will be available at:
- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Test the API

```powershell
# Run the test script to verify everything works
python test_api.py
```

## 📚 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API root information |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API documentation |

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/models/{model_name}` | Upload a model (svm, knn, rf, lr) |
| `GET` | `/models` | List all uploaded models |
| `DELETE` | `/models/{model_name}` | Delete a specific model |

### Dataset Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/datasets` | Upload a dataset (CSV or ARFF) |
| `GET` | `/datasets` | List all uploaded datasets |
| `DELETE` | `/datasets/{dataset_id}` | Delete a specific dataset |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Run predictions on a dataset |

## 🔧 Usage Examples

### 1. Upload a Model

```python
import requests

# Upload a trained model
with open('my_model.pkl', 'rb') as f:
    files = {'file': ('my_model.pkl', f, 'application/octet-stream')}
    response = requests.post('http://localhost:8000/models/rf', files=files)
    
print(response.json())
```

### 2. Upload a Dataset

```python
import requests

# Upload a CSV dataset
with open('test_data.csv', 'rb') as f:
    files = {'file': ('test_data.csv', f, 'text/csv')}
    response = requests.post('http://localhost:8000/datasets', files=files)
    
dataset_id = response.json()['dataset_id']
print(f"Dataset uploaded with ID: {dataset_id}")
```

### 3. Run Predictions

```python
import requests

# Run predictions
payload = {
    'model_name': 'rf',
    'dataset_id': 'your-dataset-id',
    'threshold': 0.5
}

response = requests.post('http://localhost:8000/predict', data=payload)
results = response.json()

print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
```

### 4. Upload File and Predict in One Step

```python
import requests

# Upload dataset file and predict simultaneously
with open('test_data.csv', 'rb') as f:
    files = {'file': ('test_data.csv', f, 'text/csv')}
    data = {'model_name': 'rf', 'threshold': 0.5}
    
    response = requests.post('http://localhost:8000/predict', files=files, data=data)
    
results = response.json()
print(f"Predictions completed: {results['total_predictions']} samples")
```

## 📊 Response Format

### Prediction Response

```json
{
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90,
    "roc_auc": 0.94,
    "pr_auc": 0.89
  },
  "predictions": [
    {
      "id": "0",
      "probability": 0.85,
      "label": 1,
      "confidence": 0.85
    }
  ],
  "feature_importance": [
    {
      "feature": "complexity",
      "importance": 0.25
    }
  ],
  "confusion_matrix": [[45, 5], [3, 47]],
  "total_predictions": 100
}
```

## 🔍 Data Preprocessing

The API automatically applies the following preprocessing steps:

1. **Data Cleaning**: Remove duplicates and empty rows/columns
2. **ID Column Removal**: Automatically detect and remove ID columns
3. **Missing Value Handling**: Impute missing values (median for numeric, mode for categorical)
4. **Categorical Encoding**: Label encoding for binary, one-hot for multi-class
5. **Feature Scaling**: StandardScaler for numerical features
6. **Feature Selection**: Optional SelectKBest for dimensionality reduction

## 🔧 Configuration

### Environment Variables

You can set the following environment variables:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info

# CORS Configuration (for production)
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Model Requirements

Uploaded models must:
- Be in pickle format (`.pkl`)
- Have `predict()` method
- Optionally have `predict_proba()` for probability estimates
- Be one of: SVM, KNN, Random Forest, or Logistic Regression

### Dataset Requirements

Datasets must:
- Be in CSV or ARFF format
- Have a clear target column (auto-detected or specify common names)
- Contain numerical or categorical features
- Be preprocessed consistently with training data

## 🚨 Error Handling

The API provides detailed error messages for common issues:

- **400 Bad Request**: Invalid file format, missing parameters
- **404 Not Found**: Model or dataset not found
- **422 Validation Error**: Invalid request format
- **500 Internal Server Error**: Server-side processing errors

## 📝 Logging

All API requests and errors are logged to:
- Console output (during development)
- `api.log` file (for production)

Log levels: DEBUG, INFO, WARNING, ERROR

## 🔒 Security Considerations

For production deployment:

1. **CORS**: Update allowed origins in production
2. **Authentication**: Add API key or JWT authentication
3. **Rate Limiting**: Implement request rate limiting
4. **File Upload**: Add file size and type restrictions
5. **Input Validation**: Enhanced input sanitization

## 🚀 Deployment

### Local Development
```bash
python start_server.py
```

### Production with Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🤝 Integration with Frontend

The API is designed to work seamlessly with your existing frontend. Key integration points:

1. **CORS**: Configured to allow frontend requests
2. **JSON Responses**: All responses in JSON format for easy parsing
3. **Error Codes**: Standard HTTP status codes for error handling
4. **File Uploads**: Support for multipart/form-data uploads

## 📞 Support

For issues or questions:
1. Check the interactive docs at `/docs`
2. Run the test script: `python test_api.py`
3. Check logs in `api.log`
4. Verify all dependencies are installed

## 📄 License

This project is part of the Software Defect Prediction research project.
