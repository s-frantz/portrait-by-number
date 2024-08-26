# add code root directory to python path

[Environment]::SetEnvironmentVariable(
    "PYTHONPATH",
    "$env:PYTHONPATH;C:\GitHub\portrait-by-number\flask",
    "User"
)

pause