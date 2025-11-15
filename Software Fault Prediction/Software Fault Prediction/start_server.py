#!/usr/bin/env python
"""
Startup script for the Software Defect Prediction API server.
Run this script to start the FastAPI server with uvicorn.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import fastapi
        import uvicorn
        import pandas
        import sklearn
        import liac_arff
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories for models and datasets."""
    models_dir = Path("models")
    datasets_dir = Path("datasets")
    
    models_dir.mkdir(exist_ok=True)
    datasets_dir.mkdir(exist_ok=True)
    
    print(f"✅ Created directories: {models_dir}, {datasets_dir}")

def start_server():
    """Start the FastAPI server with uvicorn."""
    if not check_requirements():
        return False
    
    create_directories()
    
    print("🚀 Starting Software Defect Prediction API server...")
    print("📖 API documentation will be available at: http://localhost:8000/docs")
    print("🔗 API root endpoint: http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = start_server()
    if not success:
        sys.exit(1)
