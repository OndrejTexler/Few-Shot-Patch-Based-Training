@echo off
setlocal ENABLEDELAYEDEXPANSION

call "vcvarsall.bat" x86

:compile
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
cl main_BilateralAdv.cpp /nologo /DNDEBUG /O2 /EHsc /Fe"bilateralAdv.exe" /link || goto error
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

del main_BilateralAdv.obj

goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
