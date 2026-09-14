cd @echo off
cd ..
cd ..
setlocal
set PROJECTPATH=%cd%
set PYTHONDIR-%PYTHONPATH%
echo "Python installed at: '%PYTHONDIR%'"
echo "My project path is: '%PROJECTPATH%'"
set MAINPATH=%PROJECTPATH%\streamlit_app.py
set PATH=%PYTHONDIR%;%PATH%
echo "Run command is: 'python.exe -m streamlit run "%MAINPATH%"
python.exe -m streamlit run "%MAINPATH%"
endlocal