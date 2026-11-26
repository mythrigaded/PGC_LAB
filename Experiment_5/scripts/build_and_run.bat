@echo off
REM Experiment 5 - Neural Network Layers in CUDA
REM Builds and runs src\neural_network.cu.
REM Run from the Experiment_5 folder inside a "Developer Command Prompt for VS"
REM (cl.exe must be on PATH, otherwise nvcc stops with "Cannot find compiler 'cl.exe'").
REM
REM -arch=sm_120 builds native code for compute capability 12.0 (RTX 50-series, e.g. RTX 5060 Ti).
REM For another GPU, change it (e.g. sm_89 for RTX 40-series / RTX 4500 Ada, sm_75 for a Tesla T4).

nvcc -O2 -arch=sm_120 src\neural_network.cu -o neural_network.exe
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

REM The manual asks for three runs to see how much the timings vary.
neural_network.exe
neural_network.exe
neural_network.exe
