@echo off
REM One-click setup for Windows. Double-click or run in Command Prompt inside the project folder.
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\build_corpus.py
python scripts\ingest_corpus.py --embedder hashing
echo.
echo Setup finished. Start the app with:  venv\Scripts\activate  then  streamlit run app.py
pause
