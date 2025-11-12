@echo off
echo ===== Building Project =====

set COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
set OPENCV=D:\opencv

echo Cleaning previous build...
del *.obj 2>nul
del digit_recognition.exe 2>nul

echo Step 1: Compiling main.cpp...
%COMPILER% /EHsc /I"%OPENCV%\build\include" /Isrc /c src\main.cpp
if errorlevel 1 goto error

echo Step 2: Compiling neural_network.cpp...
%COMPILER% /EHsc /I"%OPENCV%\build\include" /Isrc /c src\neural_network.cpp
if errorlevel 1 goto error

echo Step 3: Compiling image_processor.cpp...
%COMPILER% /EHsc /I"%OPENCV%\build\include" /Isrc /c src\image_processor.cpp
if errorlevel 1 goto error

echo Step 4: Compiling drawing_interface.cpp...
%COMPILER% /EHsc /I"%OPENCV%\build\include" /Isrc /c src\drawing_interface.cpp
if errorlevel 1 goto error

echo Step 5: Linking...
%COMPILER% main.obj neural_network.obj image_processor.obj drawing_interface.obj /link /LIBPATH:"%OPENCV%\build\x64\vc16\lib" opencv_world4110.lib /OUT:digit_recognition.exe
if errorlevel 1 goto error

echo Step 6: Copying DLL...
copy "%OPENCV%\build\x64\vc16\bin\opencv_world4110.dll" .

echo.
echo ===== BUILD SUCCESSFUL! =====
echo Run: digit_recognition.exe
goto end

:error
echo.
echo ===== BUILD FAILED =====

:end
pause