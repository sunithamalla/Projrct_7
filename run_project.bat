@echo off
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Training model...
python disaster_tweet_classification.py
echo.
echo Starting Flask application...
python app.py
pause
