@echo off
cd /d "%~dp0"
echo.
echo VisionX — الخادم: http://127.0.0.1:8000
echo ملاحظة: اكتب 127.0.0.1 وليس localhost اذا المتصفح ما يفتح.
echo.
start "" cmd /c "timeout /t 4 /nobreak ^>nul && start http://127.0.0.1:8000/"
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --log-level info
pause
