#!/usr/bin/env python3
"""
Setup script for Credit Card Fraud Detection System
Run this script to set up the environment and install dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['data', 'models', 'logs', 'results']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_sample_data():
    """Create sample data if no data exists"""
    data_file = Path("data/creditcard.csv")
    if not data_file.exists():
        print("📊 Creating sample data...")
        return run_command("python3 test_sample_data.py", "Creating sample dataset")
    else:
        print("✅ Data file already exists")
        return True

def run_initial_setup():
    """Run initial setup and training"""
    print("🚀 Running initial setup...")
    return run_command("python3 main.py --full", "Running full pipeline")

def main():
    """Main setup function"""
    print("🚨 Credit Card Fraud Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create sample data
    if not create_sample_data():
        print("❌ Failed to create sample data")
        sys.exit(1)
    
    # Run initial setup
    if not run_initial_setup():
        print("❌ Failed to run initial setup")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start API server: python main.py --api")
    print("2. Start dashboard: python main.py --dashboard")
    print("3. Start both: python main.py --serve")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
