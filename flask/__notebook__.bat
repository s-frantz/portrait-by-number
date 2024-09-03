:: define vars
set venv=C:\PythonVenv\portrait-by-number\Scripts\python.exe
set nbdir=%~dp0
set port=9999

:: start notebook
%venv% -m jupyter lab --notebook-dir=%nbdir% --port %port%