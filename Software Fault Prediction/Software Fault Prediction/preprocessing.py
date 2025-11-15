"""
Data preprocessing utilities for software defect prediction.
This module contains functions for data cleaning, feature engineering,
and preprocessing pipelines consistent with the training phase.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for software defect prediction.
    Handles various data formats and applies consistent preprocessing steps.
    """
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        self.feature_selector = None
        self.is_fitted = False
        
    def detect_target_column(self, df: pd.DataFrame) -> str:
        """
        Automatically detect the target column based on common naming patterns.
        """
        # Common target column names for defect prediction
        target_patterns = [
            'class', 'label', 'target', 'diagnosis', 'defect', 'bug', 
            'error', 'fault', 'issue', 'vulnerable', 'clean'
        ]
        
        for col in df.columns:
            if col.lower() in target_patterns:
                return col
                
        # If binary classification with boolean/binary values in last column
        last_col = df.columns[-1]
        unique_vals = df[last_col].unique()
        if len(unique_vals) <= 5:  # Likely categorical
            return last_col
            
        # Default to last column
        logger.warning(f"Could not auto-detect target column, using last column: {last_col}")
        return last_col
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning steps.
        """
        # Remove duplicate rows
        initial_shape = df.shape
        df = df.drop_duplicates()
        
        if df.shape[0] < initial_shape[0]:
            logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Remove columns with all missing values
        df = df.dropna(axis=1, how='all')
        
        # Remove rows with all missing values
        df = df.dropna(axis=0, how='all')
        
        return df
    
    def handle_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ID columns that are not useful for prediction.
        """
        # Common ID column patterns
        id_patterns = ['id', 'index', 'row_id', 'record_id', 'sample_id']
        
        columns_to_drop = []
        for col in df.columns:
            # Check if column name suggests it's an ID
            if col.lower() in id_patterns:
                columns_to_drop.append(col)
                continue
                
            # Check if column has unique values for each row (likely an ID)
            if df[col].nunique() == len(df) and df[col].dtype in ['int64', 'float64', 'object']:
                columns_to_drop.append(col)
                continue
        
        if columns_to_drop:
            logger.info(f"Removing ID columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
            
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        # Separate features and target
        features = df.drop(columns=[target_col])
        target = df[target_col]
        
        # Handle missing values in features
        numeric_features = features.select_dtypes(include=[np.number]).columns
        categorical_features = features.select_dtypes(include=['object']).columns
        
        # Impute numeric features with median
        if len(numeric_features) > 0:
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='median')
                features[numeric_features] = self.imputer.fit_transform(features[numeric_features])
            else:
                features[numeric_features] = self.imputer.transform(features[numeric_features])
        
        # Impute categorical features with mode
        for col in categorical_features:
            if features[col].isnull().sum() > 0:
                mode_value = features[col].mode()[0] if len(features[col].mode()) > 0 else 'unknown'
                features[col] = features[col].fillna(mode_value)
        
        # Handle missing values in target (remove rows with missing target)
        mask = ~target.isnull()
        features = features[mask]
        target = target[mask]
        
        # Combine back
        result = pd.concat([features, target], axis=1)
        return result
    
    def encode_categorical_variables(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Encode categorical variables appropriately.
        """
        # Separate features and target
        features = df.drop(columns=[target_col])
        target = df[target_col]
        
        # Encode target variable
        if target.dtype == 'object' or target.dtype.name == 'category':
            if target_col not in self.label_encoders:
                self.label_encoders[target_col] = LabelEncoder()
                target = pd.Series(self.label_encoders[target_col].fit_transform(target), 
                                 index=target.index, name=target.name)
            else:
                target = pd.Series(self.label_encoders[target_col].transform(target), 
                                 index=target.index, name=target.name)
        
        # Handle categorical features
        categorical_features = features.select_dtypes(include=['object']).columns
        
        for col in categorical_features:
            # For high cardinality categorical variables, use target encoding or drop
            if features[col].nunique() > 50:
                logger.warning(f"High cardinality column {col} with {features[col].nunique()} unique values. Dropping.")
                features = features.drop(columns=[col])
                continue
            
            # For binary categorical variables, use label encoding
            if features[col].nunique() == 2:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features[col] = self.label_encoders[col].fit_transform(features[col])
                else:
                    features[col] = self.label_encoders[col].transform(features[col])
            else:
                # For multi-class categorical variables, use one-hot encoding
                features = pd.get_dummies(features, columns=[col], prefix=col, drop_first=True)
        
        # Combine back
        result = pd.concat([features, target], axis=1)
        return result
    
    def scale_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Scale numerical features.
        """
        # Separate features and target
        features = df.drop(columns=[target_col])
        target = df[target_col]
        
        # Scale only numerical features
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 0:
            if self.scaler is None:
                self.scaler = StandardScaler()
                features[numeric_features] = self.scaler.fit_transform(features[numeric_features])
            else:
                features[numeric_features] = self.scaler.transform(features[numeric_features])
        
        # Combine back
        result = pd.concat([features, target], axis=1)
        return result
    
    def select_features(self, df: pd.DataFrame, target_col: str, 
                       k_features: Optional[int] = None) -> pd.DataFrame:
        """
        Select the most important features.
        """
        if k_features is None:
            return df
            
        # Separate features and target
        features = df.drop(columns=[target_col])
        target = df[target_col]
        
        # Apply feature selection
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
            features_selected = self.feature_selector.fit_transform(features, target)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            selected_features = features.columns[selected_mask]
            
            features_df = pd.DataFrame(features_selected, columns=selected_features, index=features.index)
        else:
            features_selected = self.feature_selector.transform(features)
            selected_mask = self.feature_selector.get_support()
            selected_features = features.columns[selected_mask]
            features_df = pd.DataFrame(features_selected, columns=selected_features, index=features.index)
        
        # Combine back
        result = pd.concat([features_df, target], axis=1)
        return result
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                     k_features: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
        """
        Fit the preprocessor and transform the data.
        """
        # Auto-detect target column if not provided
        if target_col is None:
            target_col = self.detect_target_column(df)
        
        logger.info(f"Starting preprocessing with target column: {target_col}")
        logger.info(f"Initial data shape: {df.shape}")
        
        # Apply preprocessing steps
        df = self.clean_data(df)
        df = self.handle_id_columns(df)
        df = self.handle_missing_values(df, target_col)
        df = self.encode_categorical_variables(df, target_col)
        df = self.scale_features(df, target_col)
        
        if k_features:
            df = self.select_features(df, target_col, k_features)
        
        self.is_fitted = True
        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        
        return df, target_col
    
    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        logger.info(f"Transforming data with shape: {df.shape}")
        
        # Apply preprocessing steps (without fitting)
        df = self.clean_data(df)
        df = self.handle_id_columns(df)
        df = self.handle_missing_values(df, target_col)
        df = self.encode_categorical_variables(df, target_col)
        df = self.scale_features(df, target_col)
        
        if self.feature_selector:
            df = self.select_features(df, target_col)
        
        logger.info(f"Transformation completed. Final shape: {df.shape}")
        return df

def quick_preprocess(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Quick preprocessing function for simple use cases.
    """
    preprocessor = DataPreprocessor()
    return preprocessor.fit_transform(df, target_col)

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
