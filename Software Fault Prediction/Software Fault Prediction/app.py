"""
FastAPI Backend for Software Defect Prediction
A complete API server to support frontend charts with model management,
dataset ingestion, and prediction capabilities.
"""

import os
import pickle
import joblib
import uuid
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import io
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)

# Import sklearn models to ensure they're available for unpickling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# import liac_arff  # Temporarily disabled

# Import local modules
from config import settings, MODEL_TYPE_MAPPING
from preprocessing import DataPreprocessor, split_features_target

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.TITLE,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

# Create directories for storing models and datasets (already done in settings)
MODELS_DIR = settings.MODELS_DIR
DATASETS_DIR = settings.DATASETS_DIR

# In-memory storage for datasets and models
models_storage: Dict[str, Any] = {}
datasets_storage: Dict[str, pd.DataFrame] = {}
preprocessors_storage: Dict[str, DataPreprocessor] = {}

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use (svm, knn, rf, lr)")
    dataset_id: Optional[str] = Field(None, description="ID of uploaded dataset")
    threshold: Optional[float] = Field(0.5, description="Classification threshold")

class PredictionResult(BaseModel):
    id: str
    probability: float
    label: int
    confidence: float

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]

class PredictionResponse(BaseModel):
    metrics: MetricsResponse
    predictions: List[PredictionResult]
    feature_importance: Optional[List[Dict[str, float]]]
    confusion_matrix: List[List[int]]
    total_predictions: int

class ModelInfo(BaseModel):
    model_name: str
    uploaded_at: str
    model_type: str
    file_size: int

class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    uploaded_at: str
    rows: int
    columns: int
    format: str

# Utility functions
def detect_file_format(file_content: bytes, filename: str) -> str:
    """Detect if file is CSV or ARFF format"""
    try:
        # Try to decode as text
        content_str = file_content.decode('utf-8').strip()
        
        # Check for ARFF format
        if content_str.startswith('@relation') or '@data' in content_str.lower():
            return 'arff'
        elif filename.lower().endswith('.arff'):
            return 'arff'
        else:
            return 'csv'
    except UnicodeDecodeError:
        # If can't decode as text, assume CSV
        return 'csv'

def parse_arff_file(file_content: bytes) -> pd.DataFrame:
    """Parse ARFF file and return pandas DataFrame"""
    try:
        # Temporarily disabled - only CSV support for now
        raise HTTPException(status_code=400, detail="ARFF format temporarily not supported. Please use CSV format.")
        
        # # Convert bytes to string
        # content_str = file_content.decode('utf-8')
        # 
        # # Use liac-arff to parse
        # arff_data = liac_arff.loads(content_str)
        # 
        # # Convert to DataFrame
        # df = pd.DataFrame(arff_data['data'])
        # df.columns = [attr[0] for attr in arff_data['attributes']]
        # 
        # return df
    except Exception as e:
        logger.error(f"Error parsing ARFF file: {e}")
        raise HTTPException(status_code=400, detail=f"Error parsing ARFF file: {str(e)}")

def get_feature_importance(model, feature_names: List[str]) -> List[Dict[str, float]]:
    """Extract feature importance from model if available"""
    try:
        importance = None
        
        # Different models store feature importance differently
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(model.coef_.shape) == 1:
                importance = np.abs(model.coef_)
            else:
                importance = np.abs(model.coef_[0])
        
        if importance is not None:
            return [
                {"feature": name, "importance": float(imp)} 
                for name, imp in zip(feature_names, importance)
            ]
        
        return None
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Software Defect Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "models": "/models",
            "datasets": "/datasets", 
            "predict": "/predict"
        }
    }

@app.post("/models/{model_name}")
async def upload_model(
    model_name: str,
    file: UploadFile = File(...)
):
    """Upload a serialized model (pickle format)"""
    try:
        # Validate model name
        if model_name not in settings.SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model name. Must be one of: {settings.SUPPORTED_MODELS}"
            )
        
        # Validate file format
        if not any(file.filename.endswith(fmt) for fmt in settings.SUPPORTED_FORMATS):
            raise HTTPException(
                status_code=400,
                detail=f"Model file must be in supported format: {settings.SUPPORTED_FORMATS}"
            )
          # Read and load the model
        contents = await file.read()
        
        try:
            # Ensure all sklearn modules are in namespace for pickle loading
            import sklearn
            import sklearn.neighbors
            import sklearn.svm
            import sklearn.ensemble
            import sklearn.linear_model
            import sklearn.tree
            import sklearn.naive_bayes
            import sklearn.neural_network
            import sklearn.preprocessing
            import sklearn.feature_selection
            import sklearn.pipeline
            import sklearn.model_selection
            
            # Try to load with pickle first, then joblib
            if file.filename.endswith('.pkl'):
                model = pickle.loads(contents)
            elif file.filename.endswith('.joblib'):
                # For joblib, we need to save to a temporary file first
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
                    tmp_file.write(contents)
                    tmp_file.flush()
                    model = joblib.load(tmp_file.name)
                    os.unlink(tmp_file.name)  # Clean up temp file
            else:
                model = pickle.loads(contents)  # Default to pickle
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error loading pickle file: {str(e)}"
            )
        
        # Store model in memory and on disk
        model_path = MODELS_DIR / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            f.write(contents)
        
        models_storage[model_name] = model
        
        logger.info(f"Model {model_name} uploaded successfully")
        
        return {
            "message": f"Model {model_name} uploaded successfully",
            "model_name": model_name,
            "file_size": len(contents),
            "model_type": type(model).__name__,
            "uploaded_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/datasets")
async def upload_dataset(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Upload a test dataset (CSV or ARFF format)"""
    try:
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Read file content
        contents = await file.read()
        
        # Check file size
        if len(contents) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Detect file format
        file_format = detect_file_format(contents, file.filename)
        
        # Parse based on format
        if file_format == 'arff':
            df = parse_arff_file(contents)
        else:
            # Try CSV with different separators
            try:
                df = pd.read_csv(io.BytesIO(contents))
            except:
                try:
                    df = pd.read_csv(io.BytesIO(contents), sep=';')
                except:
                    df = pd.read_csv(io.BytesIO(contents), sep='\t')
        
        # Basic validation
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        # Store dataset
        datasets_storage[dataset_id] = df
        
        # Save to disk
        dataset_path = DATASETS_DIR / f"{dataset_id}.csv"
        df.to_csv(dataset_path, index=False)
        
        logger.info(f"Dataset {dataset_id} uploaded successfully. Shape: {df.shape}")
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "format": file_format,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "uploaded_at": datetime.now().isoformat(),
            "message": "Dataset uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    model_name: str = Form(...),
    dataset_id: Optional[str] = Form(None),
    threshold: float = Form(settings.DEFAULT_CLASSIFICATION_THRESHOLD),
    file: Optional[UploadFile] = File(None)
):
    """Run predictions using specified model and dataset"""
    try:
        # Validate model exists
        if model_name not in models_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found. Please upload the model first."
            )
        
        model = models_storage[model_name]
        
        # Get dataset
        if dataset_id and dataset_id in datasets_storage:
            df = datasets_storage[dataset_id].copy()
        elif file:
            # Upload and process file on the fly
            contents = await file.read()
            
            # Check file size
            if len(contents) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            file_format = detect_file_format(contents, file.filename)
            
            if file_format == 'arff':
                df = parse_arff_file(contents)
            else:
                try:
                    df = pd.read_csv(io.BytesIO(contents))
                except:
                    try:
                        df = pd.read_csv(io.BytesIO(contents), sep=';')
                    except:
                        df = pd.read_csv(io.BytesIO(contents), sep='\t')
        else:
            raise HTTPException(
                status_code=400,
                detail="Either dataset_id or file must be provided"
            )
        
        # Preprocess dataset
        preprocessor = DataPreprocessor()
        df_processed, target_col = preprocessor.fit_transform(df)
        
        # Extract features and target
        X, y_true = split_features_target(df_processed, target_col)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            if y_proba.shape[1] == 2:
                # Binary classification
                probabilities = y_proba[:, 1]
            else:
                # Multi-class - use max probability
                probabilities = np.max(y_proba, axis=1)
        else:
            # Use prediction confidence (distance from decision boundary for SVM)
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X)
                probabilities = 1 / (1 + np.exp(-decision_scores))  # Sigmoid
            else:
                probabilities = np.ones(len(y_pred)) * 0.5  # Default confidence
        
        # Apply threshold for binary classification
        if threshold != 0.5 and hasattr(model, 'predict_proba'):
            y_pred_thresh = (probabilities >= threshold).astype(int)
        else:
            y_pred_thresh = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_thresh)
        precision = precision_score(y_true, y_pred_thresh, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred_thresh, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred_thresh, average='binary', zero_division=0)
        
        # ROC AUC (only for binary classification with probabilities)
        roc_auc = None
        pr_auc = None
        if hasattr(model, 'predict_proba') and len(np.unique(y_true)) == 2:
            try:
                roc_auc = roc_auc_score(y_true, probabilities)
                pr_auc = average_precision_score(y_true, probabilities)
            except Exception as e:
                logger.warning(f"Could not calculate AUC scores: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_thresh).tolist()
        
        # Feature importance
        feature_importance = get_feature_importance(model, list(X.columns))
        
        # Prepare predictions list
        predictions = []
        for i, (pred, prob) in enumerate(zip(y_pred_thresh, probabilities)):
            predictions.append(PredictionResult(
                id=str(i),
                probability=float(prob),
                label=int(pred),
                confidence=float(prob) if pred == 1 else float(1 - prob)
            ))
        
        # Prepare response
        metrics = MetricsResponse(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc
        )
        
        response = PredictionResponse(
            metrics=metrics,
            predictions=predictions,
            feature_importance=feature_importance,
            confusion_matrix=cm,
            total_predictions=len(predictions)
        )
        
        logger.info(f"Prediction completed for model {model_name}. Accuracy: {accuracy:.4f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/models")
async def list_models():
    """List all uploaded models"""
    models_info = []
    
    for model_name in models_storage.keys():
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if model_path.exists():
            stat = model_path.stat()
            models_info.append({
                "model_name": model_name,
                "file_size": stat.st_size,
                "model_type": type(models_storage[model_name]).__name__,
                "uploaded_at": stat.st_mtime
            })
    
    return {"models": models_info, "total": len(models_info)}

@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    datasets_info = []
    
    for dataset_id, df in datasets_storage.items():
        dataset_path = DATASETS_DIR / f"{dataset_id}.csv"
        stat = dataset_path.stat() if dataset_path.exists() else None
        
        datasets_info.append({
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": len(df.columns),
            "uploaded_at": stat.st_mtime if stat else None,
            "file_size": stat.st_size if stat else None
        })
    
    return {"datasets": datasets_info, "total": len(datasets_info)}

@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a specific model"""
    if model_name not in models_storage:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Remove from memory
    del models_storage[model_name]
    
    # Remove from disk
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if model_path.exists():
        model_path.unlink()
    
    logger.info(f"Model {model_name} deleted successfully")
    return {"message": f"Model {model_name} deleted successfully"}

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a specific dataset"""
    if dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Remove from memory
    del datasets_storage[dataset_id]
    
    # Remove from disk
    dataset_path = DATASETS_DIR / f"{dataset_id}.csv"
    if dataset_path.exists():
        dataset_path.unlink()
    
    logger.info(f"Dataset {dataset_id} deleted successfully")
    return {"message": f"Dataset {dataset_id} deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models_storage),
        "datasets_loaded": len(datasets_storage)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL
    )
