@echo off
echo Starting GelVision...
python -m pip install -r requirements.txt
python -m streamlit run gelvision/app.py
pause
