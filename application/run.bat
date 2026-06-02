@echo off
REM Launch the multimodal emotion app with UTF-8 forced.
REM Required because the NLP model has a Turkish vocabulary that breaks
REM under Windows default cp1254 encoding.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

py -3.13 -X utf8 main.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
