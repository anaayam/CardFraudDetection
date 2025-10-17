"""
Main Application for Credit Card Fraud Detection
Complete pipeline from data preprocessing to model training and API deployment
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'fastapi', 'uvicorn', 'streamlit', 'plotly',  # xgboost removed due to OpenMP dependency
        'requests', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs', 'results']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def run_data_preprocessing():
    """Run data preprocessing"""
    print("\n🔧 Running data preprocessing...")
    
    try:
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Check if data file exists
        data_file = "data/creditcard.csv"
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            print("Please download the Credit Card Fraud Detection dataset from Kaggle")
            print("and place it in the 'data/' directory")
            return False
        
        # Run preprocessing
        processed_data = preprocessor.preprocess_pipeline(data_file)
        
        if processed_data:
            print("✅ Data preprocessing completed successfully")
            return True
        else:
            print("❌ Data preprocessing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in data preprocessing: {str(e)}")
        return False

def run_model_training():
    """Run model training"""
    print("\n🤖 Training machine learning models...")
    
    try:
        from data_preprocessing import DataPreprocessor
        from model_training import FraudDetectionModels
        
        # Load preprocessed data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_pipeline("data/creditcard.csv")
        
        if not processed_data:
            print("❌ No preprocessed data found")
            return False
        
        # Train models
        fraud_models = FraudDetectionModels()
        fraud_models.train_all_models(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test']
        )
        
        # Save models
        fraud_models.save_models()
        
        print("✅ Model training completed successfully")
        print(f"Best model: {fraud_models.best_model.__class__.__name__}")
        print(f"Best F1-Score: {fraud_models.best_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting FastAPI server...")
    
    try:
        # Start the server in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "fastapi_service:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        print("✅ FastAPI server started on http://localhost:8000")
        print("📖 API documentation available at http://localhost:8000/docs")
        
        return process
        
    except Exception as e:
        print(f"❌ Error starting API server: {str(e)}")
        return None

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("\n📊 Starting Streamlit dashboard...")
    
    try:
        # Start the dashboard
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        
        # Wait a moment for dashboard to start
        time.sleep(3)
        
        print("✅ Streamlit dashboard started on http://localhost:8501")
        
        return process
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {str(e)}")
        return None

def run_full_pipeline():
    """Run the complete fraud detection pipeline"""
    print("🔍 Credit Card Fraud Detection - Full Pipeline")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup directories
    setup_directories()
    
    # Run data preprocessing
    if not run_data_preprocessing():
        return False
    
    # Run model training
    if not run_model_training():
        return False
    
    print("\n✅ Full pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Start API server: python main.py --api")
    print("2. Start dashboard: python main.py --dashboard")
    print("3. Start both: python main.py --serve")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection System")
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing only")
    parser.add_argument("--train", action="store_true", help="Run model training only")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")
    parser.add_argument("--dashboard", action="store_true", help="Start Streamlit dashboard")
    parser.add_argument("--serve", action="store_true", help="Start both API and dashboard")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--check", action="store_true", help="Check dependencies and setup")
    
    args = parser.parse_args()
    
    if args.check:
        check_dependencies()
        setup_directories()
        return
    
    if args.preprocess:
        run_data_preprocessing()
    elif args.train:
        run_model_training()
    elif args.api:
        start_api_server()
    elif args.dashboard:
        start_dashboard()
    elif args.serve:
        api_process = start_api_server()
        dashboard_process = start_dashboard()
        
        if api_process and dashboard_process:
            print("\n🎉 Both services are running!")
            print("📊 Dashboard: http://localhost:8501")
            print("🔗 API: http://localhost:8000")
            print("📖 API Docs: http://localhost:8000/docs")
            print("\nPress Ctrl+C to stop both services")
            
            try:
                # Keep the script running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping services...")
                api_process.terminate()
                dashboard_process.terminate()
                print("✅ Services stopped")
    elif args.full:
        run_full_pipeline()
    else:
        print("Credit Card Fraud Detection System")
        print("=" * 40)
        print("Available commands:")
        print("  --check      Check dependencies and setup")
        print("  --preprocess Run data preprocessing")
        print("  --train      Train machine learning models")
        print("  --api        Start FastAPI server")
        print("  --dashboard  Start Streamlit dashboard")
        print("  --serve      Start both API and dashboard")
        print("  --full       Run complete pipeline")
        print("\nExample usage:")
        print("  python main.py --full")
        print("  python main.py --serve")

if __name__ == "__main__":
    main()
