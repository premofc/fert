@echo off
echo ========================================
echo Fertilizer Recommendation System Setup
echo ========================================
echo.

echo Step 1: Installing required packages...
pip install Flask scikit-learn pandas numpy joblib
echo.

echo Step 2: Checking if model files exist...
if exist classifier.pkl (
    echo classifier.pkl found
) else (
    echo classifier.pkl NOT FOUND - will generate it
)

if exist fertilizer.pkl (
    echo fertilizer.pkl found
) else (
    echo fertilizer.pkl NOT FOUND - will generate it
)
echo.

echo Step 3: Generating/Regenerating model files...
python retrain.py
echo.

echo Step 4: Starting Flask application...
echo Server will be available at http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.
python main.py
