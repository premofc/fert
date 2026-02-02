@echo off
echo Installing required packages...
python -m pip install Flask scikit-learn pandas numpy joblib
echo.
echo Testing installation...
python -c "import sklearn; print('scikit-learn installed:', sklearn.__version__)"
echo.
echo Checking models...
python check_models.py
pause
