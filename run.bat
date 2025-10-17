@echo off
REM Credit Card Fraud Detection - Windows Run Script

echo === Credit Card Fraud Detection System ===

REM Check if Java is installed
java -version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Java is not installed or not in PATH
    echo Please install Java 11 or higher
    pause
    exit /b 1
)

REM Check if Maven is installed
mvn -version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Maven is not installed or not in PATH
    echo Please install Maven 3.6 or higher
    pause
    exit /b 1
)

REM Compile the project
echo Compiling project...
mvn clean compile

if %errorlevel% neq 0 (
    echo Error: Compilation failed
    pause
    exit /b 1
)

REM Check if data file exists
if not exist "data\creditcard.csv" (
    echo Warning: Data file not found at data\creditcard.csv
    echo Running demo with sample data...
    mvn exec:java -Dexec.mainClass="com.frauddetection.DemoApp"
) else (
    echo Running full application...
    mvn exec:java -Dexec.mainClass="com.frauddetection.FraudDetectionApp"
)

pause
