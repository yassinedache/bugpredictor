"""
Configuration settings for the Software Defect Prediction API.
"""

import os
from pathlib import Path
from typing import List

class Settings:
    """API configuration settings."""
    
    # Server Configuration
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # API Information
    TITLE: str = "Software Defect Prediction API"
    DESCRIPTION: str = "API for uploading ML models, datasets, and running predictions for software defect detection"
    VERSION: str = "1.0.0"
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", 
        "*"  # For development; restrict in production
    ).split(",")
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    ALLOW_CREDENTIALS: bool = True    # File Storage
    BASE_DIR: Path = Path(__file__).parent
    MODELS_DIR: Path = BASE_DIR / "trained_models"
    DATASETS_DIR: Path = BASE_DIR / "datasets"
    RESULTS_DIR: Path = BASE_DIR / "results"
    ALGORITHMS_DIR: Path = BASE_DIR / "algorithms"
    
    # Ensure directories exist
    def __post_init__(self):
        for directory in [self.MODELS_DIR, self.DATASETS_DIR, self.RESULTS_DIR, self.ALGORITHMS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    # Model Configuration
    SUPPORTED_MODELS: List[str] = [
        # KNN models
        "knn", "knn_1", "knn_2", "knn_3", "knn_4",
        # SVM models  
        "svm", "svm_1", "svm_2", "svm_3", "svm_4",
        # Random Forest models
        "rf", "rf_1", "rf_2", "rf_3", "rf_4",
        # Logistic Regression models
        "lr", "lr_1", "lr_2", "lr_3", "lr_4"
    ]
    SUPPORTED_FORMATS: List[str] = [".pkl", ".joblib"]
    
    # Dataset Configuration
    SUPPORTED_DATASET_FORMATS: List[str] = [".csv", ".arff"]
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB default
    
    # Preprocessing Configuration
    DEFAULT_IMPUTATION_STRATEGY: str = "median"
    DEFAULT_SCALING_METHOD: str = "standard"
    MAX_CATEGORICAL_CARDINALITY: int = 50
    
    # Feature Selection
    DEFAULT_FEATURE_SELECTION_RATIO: float = 0.8  # Keep 80% of features by default
    FEATURE_SELECTION_METHOD: str = "mutual_info_classif"
    
    # Prediction Configuration
    DEFAULT_CLASSIFICATION_THRESHOLD: float = 0.5
    
    # Logging Configuration
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = os.getenv("LOG_FILE", "api.log")
    
    # Security Configuration
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "100"))
    
    def __init__(self):
        """Initialize settings and create necessary directories."""
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.DATASETS_DIR.mkdir(exist_ok=True)

# Global settings instance
settings = Settings()

# Model type mapping for better organization
MODEL_TYPE_MAPPING = {
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors", 
    "rf": "Random Forest",
    "lr": "Logistic Regression"
}

# Common target column names for auto-detection
TARGET_COLUMN_PATTERNS = [
    "class", "label", "target", "diagnosis", "defect", "bug",
    "error", "fault", "issue", "vulnerable", "clean", "outcome"
]

# File format MIME types
MIME_TYPES = {
    ".pkl": "application/octet-stream",
    ".joblib": "application/octet-stream", 
    ".csv": "text/csv",
    ".arff": "text/plain"
}
