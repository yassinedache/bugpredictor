# Workspace Organization Script
# Run this script to organize the Software Defect Prediction workspace

Write-Host "Organizing Software Defect Prediction Workspace..." -ForegroundColor Green

# Navigate to main directory
Set-Location "c:\Users\dell\Desktop\Interface_sdp\Interface"

# Create organized structure
Write-Host "Creating directory structure..." -ForegroundColor Yellow

$directories = @(
    "algorithms\knn",
    "algorithms\svm", 
    "algorithms\random_forest",
    "algorithms\logistic_regression",
    "trained_models\knn",
    "trained_models\svm",
    "trained_models\rf", 
    "trained_models\lr",
    "results\knn",
    "results\svm",
    "results\comparisons",
    "notebooks",
    "scripts",
    "docs",
    "frontend\assets",
    "frontend\styles", 
    "frontend\pages"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path ".\$dir" | Out-Null
    Write-Host "Created: $dir" -ForegroundColor Green
}

# Move frontend files
Write-Host "Moving frontend files..." -ForegroundColor Yellow
if (Test-Path "*.html") { Move-Item "*.html" ".\frontend\pages\" -Force }
if (Test-Path "*.css") { Move-Item "*.css" ".\frontend\styles\" -Force }
if (Test-Path "*.svg") { Move-Item "*.svg" ".\frontend\assets\" -Force }
if (Test-Path "*.png") { Move-Item "*.png" ".\frontend\assets\" -Force }

# Copy algorithms
Write-Host "Copying algorithm files..." -ForegroundColor Yellow
if (Test-Path "c:\Users\dell\Desktop\KNN") {
    Copy-Item "c:\Users\dell\Desktop\KNN\*.py" ".\algorithms\knn\" -Force -ErrorAction SilentlyContinue
    Copy-Item "c:\Users\dell\Desktop\KNN\*.ipynb" ".\notebooks\" -Force -ErrorAction SilentlyContinue
    Write-Host "Copied KNN algorithms" -ForegroundColor Green
}

if (Test-Path "c:\Users\dell\Desktop\SVM") {
    Copy-Item "c:\Users\dell\Desktop\SVM\*.py" ".\algorithms\svm\" -Force -ErrorAction SilentlyContinue
    Write-Host "Copied SVM algorithms" -ForegroundColor Green
}

# Copy trained models
Write-Host "Copying trained models..." -ForegroundColor Yellow
if (Test-Path "c:\Users\dell\Desktop\ML_Project") {
    Copy-Item "c:\Users\dell\Desktop\ML_Project\*.pkl" ".\trained_models\" -Force -ErrorAction SilentlyContinue
    Copy-Item "c:\Users\dell\Desktop\ML_Project\*.py" ".\scripts\" -Force -ErrorAction SilentlyContinue
    Copy-Item "c:\Users\dell\Desktop\ML_Project\*.ipynb" ".\notebooks\" -Force -ErrorAction SilentlyContinue
    Write-Host "Copied ML_Project files" -ForegroundColor Green
}

# Copy results
Write-Host "Copying results..." -ForegroundColor Yellow
if (Test-Path "c:\Users\dell\Desktop\RESULTS_KNN\knn_ga_combined\RESULTS") {
    Copy-Item "c:\Users\dell\Desktop\RESULTS_KNN\knn_ga_combined\RESULTS\*" ".\results\knn\" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Copied KNN results" -ForegroundColor Green
}

# Copy datasets
Write-Host "Copying datasets..." -ForegroundColor Yellow
if (Test-Path "c:\Users\dell\Desktop\ML_Project\Dataset_reduced") {
    Copy-Item "c:\Users\dell\Desktop\ML_Project\Dataset_reduced\*" ".\datasets\" -Force -ErrorAction SilentlyContinue
    Write-Host "Copied datasets" -ForegroundColor Green
}

# Clean up duplicate venv
Write-Host "Cleaning up..." -ForegroundColor Yellow
if (Test-Path ".\venv") {
    Remove-Item ".\venv" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Removed duplicate venv" -ForegroundColor Green
}

Write-Host "Workspace organization complete!" -ForegroundColor Green
Write-Host "Main directory: c:\Users\dell\Desktop\Interface_sdp\Interface" -ForegroundColor Cyan
Write-Host "To start the server: .\venv_interface\Scripts\Activate.ps1; python app.py" -ForegroundColor Cyan
