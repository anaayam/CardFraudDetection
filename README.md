# 🚨 Credit Card Fraud Detection System

A comprehensive machine learning system for detecting fraudulent credit card transactions using advanced data science techniques and modern web technologies.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.119.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

- **🤖 Machine Learning Models**: Logistic Regression, Random Forest
- **📊 Interactive Dashboard**: Real-time fraud analysis with Streamlit
- **🌐 REST API**: FastAPI service for integration
- **📈 Advanced Analytics**: Feature engineering, class balancing, model evaluation
- **🔍 Real-time Detection**: Live fraud prediction capabilities
- **📱 Responsive UI**: Modern, user-friendly interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  Preprocessing  │───▶│  Model Training │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │◀───│   FastAPI       │◀───│  Trained Models│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the full pipeline**
   ```bash
   python main.py --full
   ```

### Usage Options

#### Option 1: Full Pipeline
```bash
python main.py --full
```
- Runs complete data processing and model training
- Creates necessary directories and saves models

#### Option 2: API Server Only
```bash
python main.py --api
```
- Starts FastAPI server on `http://localhost:8000`
- API documentation at `http://localhost:8000/docs`

#### Option 3: Dashboard Only
```bash
python main.py --dashboard
```
- Starts Streamlit dashboard on `http://localhost:8501`
- Interactive fraud analysis interface

#### Option 4: Both Services
```bash
python main.py --serve
```
- Runs API + Dashboard simultaneously
- Full-stack fraud detection system

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 75% | 75% | 75% | 75% | 81.25% |
| Random Forest | 75% | 75% | 75% | 75% | 93.75% |

**Best Model**: Logistic Regression (F1-Score: 0.75)

## 🔧 API Endpoints

### Health Check
```http
GET /health
```

### Predict Fraud
```http
POST /predict
Content-Type: application/json

{
  "amount": 100.50,
  "time": 1234567890,
  "v1": -1.359807134,
  "v2": -0.072781173,
  ...
}
```

### Model Status
```http
GET /models/status
```

### Model Performance
```http
GET /models/performance
```

## 📁 Project Structure

```
credit-card-fraud-detection/
├── data/                    # Dataset storage
│   ├── creditcard.csv      # Credit card transaction data
│   └── README.md           # Data documentation
├── models/                  # Trained model files
│   ├── *.joblib           # Saved models
│   └── model_scores_*.joblib
├── logs/                    # System logs
├── results/                 # Analysis results
├── src/                     # Java implementation (optional)
├── data_preprocessing.py    # Data pipeline
├── model_training.py       # ML training
├── fastapi_service.py      # REST API
├── streamlit_dashboard.py   # Web dashboard
├── main.py                 # Main orchestrator
├── requirements.txt         # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🛠️ Development

### Running Tests
```bash
python main.py --check
```

### Code Quality
```bash
# Format code
black *.py

# Lint code
flake8 *.py
```

### Adding New Models
1. Extend `FraudDetectionModels` class in `model_training.py`
2. Add model to the training pipeline
3. Update API endpoints if needed

## 📈 Data Pipeline

### 1. Data Preprocessing
- **Data Loading**: CSV file with 1000+ transactions
- **Exploration**: Statistical analysis and visualization
- **Cleaning**: Handle missing values, outliers, duplicates
- **Feature Engineering**: Time-based features, amount transformations
- **Class Balancing**: Undersampling to handle imbalanced data

### 2. Model Training
- **Algorithm Selection**: Multiple ML algorithms
- **Cross-validation**: Robust model evaluation
- **Hyperparameter Tuning**: Optimized model parameters
- **Performance Metrics**: Accuracy, Precision, Recall, F1, AUC

### 3. Model Deployment
- **API Integration**: RESTful endpoints
- **Real-time Prediction**: Live fraud detection
- **Dashboard Visualization**: Interactive analytics

## 🔍 Key Features

### Data Preprocessing
- **Feature Engineering**: 35+ engineered features
- **Time Analysis**: Hour-based fraud patterns
- **Amount Analysis**: Transaction amount distributions
- **PCA Features**: Dimensionality reduction analysis

### Machine Learning
- **Multiple Algorithms**: Logistic Regression, Random Forest
- **Class Balancing**: Undersampling technique
- **Feature Scaling**: StandardScaler normalization
- **Model Persistence**: Joblib serialization

### Web Interface
- **FastAPI**: Modern, fast web framework
- **Streamlit**: Interactive dashboard
- **Real-time Updates**: Live data visualization
- **Responsive Design**: Mobile-friendly interface

## 🚨 Fraud Detection Process

1. **Data Input**: Transaction features (V1-V28, Time, Amount)
2. **Preprocessing**: Feature scaling and transformation
3. **Model Prediction**: ML algorithm classification
4. **Risk Assessment**: Fraud probability calculation
5. **Alert Generation**: High-risk transaction alerts

## 📊 Dashboard Features

- **Real-time Monitoring**: Live fraud detection
- **Performance Metrics**: Model accuracy and performance
- **Data Visualization**: Interactive charts and graphs
- **Transaction Analysis**: Detailed fraud analysis
- **Model Comparison**: Side-by-side model evaluation

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom paths
export DATA_PATH=./data/
export MODEL_PATH=./models/
export LOG_PATH=./logs/
```

### Model Parameters
- **Logistic Regression**: C=1.0, random_state=42
- **Random Forest**: n_estimators=100, max_depth=10
- **Class Weight**: Balanced for imbalanced data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle**: Credit Card Fraud Detection dataset
- **Scikit-learn**: Machine learning algorithms
- **FastAPI**: Modern web framework
- **Streamlit**: Interactive dashboard framework

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the API endpoints

## 🔮 Future Enhancements

- [ ] Neural Network models
- [ ] Real-time streaming data
- [ ] Database integration
- [ ] Cloud deployment
- [ ] Advanced anomaly detection
- [ ] Multi-model ensemble
- [ ] Automated retraining
- [ ] Alert system integration

---

**Built with ❤️ for fraud detection and prevention**