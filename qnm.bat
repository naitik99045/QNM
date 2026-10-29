@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0src;%PYTHONPATH%
python -m qnm.cli.main %*
