@echo off
setlocal
cd /d "%~dp0"
python upload_models_to_huggingface.py
pause
