@echo off
cd /d "%~dp0"
start "PowerNZ Beta" http://127.0.0.1:8000
python -m web
