@echo off
setlocal ENABLEDELAYEDEXPANSION

call "vcvarsall.bat" amd64

:compile
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
cl disflow.cpp /I"opencv-4.2.0\include" /Fe"disflow.exe" /wd4312 /MT /O2 /DNDEBUG /EHsc /link /LIBPATH:"opencv-4.2.0\lib" opencv_world420.lib || goto error
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

del disflow.obj

goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
