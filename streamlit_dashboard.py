"""
Streamlit Dashboard for Credit Card Fraud Detection
Interactive dashboard for visualizing fraud detection results and model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .normal-transaction {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

class FraudDetectionDashboard:
    def __init__(self):
        self.api_url = API_BASE_URL
        self.session = requests.Session()
    
    def check_api_health(self):
        """Check if the API is running and healthy"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200, response.json()
        except:
            return False, None
    
    def get_model_info(self):
        """Get information about available models"""
        try:
            response = self.session.get(f"{self.api_url}/models", timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def get_model_performance(self):
        """Get model performance metrics"""
        try:
            response = self.session.get(f"{self.api_url}/models/performance", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_fraud_stats(self):
        """Get fraud detection statistics"""
        try:
            response = self.session.get(f"{self.api_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def predict_fraud(self, transaction_data):
        """Predict fraud for a transaction"""
        try:
            response = self.session.post(
                f"{self.api_url}/predict",
                json=transaction_data,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

# Initialize dashboard
dashboard = FraudDetectionDashboard()

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Credit Card Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy, health_data = dashboard.check_api_health()
    
    if not api_healthy:
        st.error("🚨 API is not running! Please start the FastAPI service first.")
        st.info("Run: `python fastapi_service.py` or `uvicorn fastapi_service:app --reload`")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Model Performance", "Fraud Prediction", "Data Analysis", "API Status"]
    )
    
    # Display selected page
    if page == "Overview":
        show_overview(health_data)
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Fraud Prediction":
        show_fraud_prediction()
    elif page == "Data Analysis":
        show_data_analysis()
    elif page == "API Status":
        show_api_status(health_data)

def show_overview(health_data):
    """Show overview dashboard"""
    st.header("📊 Overview")
    
    # API Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Status", "🟢 Healthy" if health_data else "🔴 Down")
    
    with col2:
        models_count = len(health_data.get('available_models', [])) if health_data else 0
        st.metric("Models Loaded", models_count)
    
    with col3:
        st.metric("API Version", health_data.get('api_version', 'Unknown') if health_data else 'N/A')
    
    with col4:
        timestamp = health_data.get('timestamp', 'Unknown') if health_data else 'N/A'
        st.metric("Last Updated", timestamp[:19] if timestamp != 'Unknown' else 'N/A')
    
    # Get fraud statistics
    fraud_stats = dashboard.get_fraud_stats()
    
    if fraud_stats:
        st.subheader("🎯 Model Performance Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Accuracy", f"{fraud_stats['average_metrics']['avg_accuracy']:.3f}")
            st.metric("Best Precision", f"{fraud_stats['average_metrics']['best_precision']:.3f}")
        
        with col2:
            st.metric("Best Recall", f"{fraud_stats['average_metrics']['best_recall']:.3f}")
            st.metric("Best F1-Score", f"{fraud_stats['average_metrics']['best_f1']:.3f}")
        
        # Model comparison chart
        st.subheader("📈 Model Performance Comparison")
        
        model_performance = dashboard.get_model_performance()
        if model_performance and 'model_performance' in model_performance:
            performance_data = model_performance['model_performance']
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, metrics in performance_data.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'AUC': metrics['auc']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Plot comparison
            fig = px.bar(
                df_comparison.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                x='Model',
                y='Score',
                color='Metric',
                title='Model Performance Comparison',
                barmode='group'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No model performance data available. Please train models first.")

def show_model_performance():
    """Show detailed model performance"""
    st.header("🎯 Model Performance Analysis")
    
    model_performance = dashboard.get_model_performance()
    
    if not model_performance or 'model_performance' not in model_performance:
        st.warning("No model performance data available.")
        return
    
    performance_data = model_performance['model_performance']
    
    # Create performance DataFrame
    performance_df = pd.DataFrame([
        {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'AUC': metrics['auc']
        }
        for model_name, metrics in performance_data.items()
    ])
    
    # Display metrics table
    st.subheader("📊 Performance Metrics")
    st.dataframe(performance_df.round(4), use_container_width=True)
    
    # Performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve simulation (since we don't have actual curves from API)
        st.subheader("📈 Accuracy vs Precision")
        fig = px.scatter(
            performance_df,
            x='Accuracy',
            y='Precision',
            size='AUC',
            color='Model',
            title='Accuracy vs Precision (size = AUC)',
            hover_data=['F1-Score', 'Recall']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # F1-Score comparison
        fig = px.bar(
            performance_df,
            x='Model',
            y='F1-Score',
            title='F1-Score Comparison',
            color='F1-Score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📈 Detailed Performance Metrics")
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=metrics_to_plot,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
    )
    
    for i, metric in enumerate(metrics_to_plot):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        if metric == 'AUC':
            fig.add_trace(
                go.Scatter(
                    x=performance_df['Model'],
                    y=performance_df[metric],
                    mode='markers+lines',
                    name=metric,
                    marker=dict(size=10)
                ),
                row=row, col=col
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=performance_df['Model'],
                    y=performance_df[metric],
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_fraud_prediction():
    """Show fraud prediction interface"""
    st.header("🔮 Fraud Prediction")
    
    st.subheader("Enter Transaction Details")
    
    # Create form for transaction input
    with st.form("fraud_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time (seconds)", value=0.0, step=1.0)
            amount = st.number_input("Amount", value=0.0, step=0.01, format="%.2f")
            
            # V1-V14 features
            v1 = st.number_input("V1", value=0.0, step=0.01, format="%.6f")
            v2 = st.number_input("V2", value=0.0, step=0.01, format="%.6f")
            v3 = st.number_input("V3", value=0.0, step=0.01, format="%.6f")
            v4 = st.number_input("V4", value=0.0, step=0.01, format="%.6f")
            v5 = st.number_input("V5", value=0.0, step=0.01, format="%.6f")
            v6 = st.number_input("V6", value=0.0, step=0.01, format="%.6f")
            v7 = st.number_input("V7", value=0.0, step=0.01, format="%.6f")
            v8 = st.number_input("V8", value=0.0, step=0.01, format="%.6f")
            v9 = st.number_input("V9", value=0.0, step=0.01, format="%.6f")
            v10 = st.number_input("V10", value=0.0, step=0.01, format="%.6f")
            v11 = st.number_input("V11", value=0.0, step=0.01, format="%.6f")
            v12 = st.number_input("V12", value=0.0, step=0.01, format="%.6f")
            v13 = st.number_input("V13", value=0.0, step=0.01, format="%.6f")
            v14 = st.number_input("V14", value=0.0, step=0.01, format="%.6f")
        
        with col2:
            # V15-V28 features
            v15 = st.number_input("V15", value=0.0, step=0.01, format="%.6f")
            v16 = st.number_input("V16", value=0.0, step=0.01, format="%.6f")
            v17 = st.number_input("V17", value=0.0, step=0.01, format="%.6f")
            v18 = st.number_input("V18", value=0.0, step=0.01, format="%.6f")
            v19 = st.number_input("V19", value=0.0, step=0.01, format="%.6f")
            v20 = st.number_input("V20", value=0.0, step=0.01, format="%.6f")
            v21 = st.number_input("V21", value=0.0, step=0.01, format="%.6f")
            v22 = st.number_input("V22", value=0.0, step=0.01, format="%.6f")
            v23 = st.number_input("V23", value=0.0, step=0.01, format="%.6f")
            v24 = st.number_input("V24", value=0.0, step=0.01, format="%.6f")
            v25 = st.number_input("V25", value=0.0, step=0.01, format="%.6f")
            v26 = st.number_input("V26", value=0.0, step=0.01, format="%.6f")
            v27 = st.number_input("V27", value=0.0, step=0.01, format="%.6f")
            v28 = st.number_input("V28", value=0.0, step=0.01, format="%.6f")
        
        # Sample data buttons
        st.subheader("Quick Test with Sample Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("🧪 Test Normal Transaction"):
                # Sample normal transaction
                sample_data = {
                    "Time": 0.0,
                    "V1": -0.073403, "V2": 0.238, "V3": -0.073403, "V4": 0.238, "V5": -0.073403,
                    "V6": 0.238, "V7": -0.073403, "V8": 0.238, "V9": -0.073403, "V10": 0.238,
                    "V11": -0.073403, "V12": 0.238, "V13": -0.073403, "V14": 0.238, "V15": -0.073403,
                    "V16": 0.238, "V17": -0.073403, "V18": 0.238, "V19": -0.073403, "V20": 0.238,
                    "V21": -0.073403, "V22": 0.238, "V23": -0.073403, "V24": 0.238, "V25": -0.073403,
                    "V26": 0.238, "V27": -0.073403, "V28": 0.238,
                    "Amount": 149.62
                }
                prediction = dashboard.predict_fraud(sample_data)
                if prediction:
                    display_prediction_result(prediction)
        
        with col2:
            if st.form_submit_button("🚨 Test Fraudulent Transaction"):
                # Sample fraudulent transaction
                sample_data = {
                    "Time": 0.0,
                    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155, "V5": -0.338262,
                    "V6": 0.462388, "V7": 0.239599, "V8": 0.098698, "V9": 0.363787, "V10": 0.090794,
                    "V11": -0.551601, "V12": -0.617801, "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
                    "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
                    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928, "V25": 0.128539,
                    "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
                    "Amount": 2.69
                }
                prediction = dashboard.predict_fraud(sample_data)
                if prediction:
                    display_prediction_result(prediction)
        
        # Manual prediction
        if st.form_submit_button("🔍 Predict Fraud"):
            transaction_data = {
                "Time": time,
                "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6, "V7": v7, "V8": v8,
                "V9": v9, "V10": v10, "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
                "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20, "V21": v21, "V22": v22,
                "V23": v23, "V24": v24, "V25": v25, "V26": v26, "V27": v27, "V28": v28,
                "Amount": amount
            }
            
            prediction = dashboard.predict_fraud(transaction_data)
            if prediction:
                display_prediction_result(prediction)
            else:
                st.error("Failed to get prediction. Please check API connection.")

def display_prediction_result(prediction):
    """Display prediction result"""
    st.subheader("🎯 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction['is_fraud']:
            st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
            st.error("🚨 FRAUD DETECTED!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="normal-transaction">', unsafe_allow_html=True)
            st.success("✅ Normal Transaction")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Fraud Probability", f"{prediction['fraud_probability']:.3f}")
        st.metric("Risk Level", prediction['risk_level'])
    
    with col3:
        st.metric("Confidence", f"{prediction['confidence']:.3f}")
        st.metric("Model Used", prediction['model_used'])
    
    # Probability visualization
    st.subheader("📊 Probability Distribution")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction['fraud_probability'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Probability"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def show_data_analysis():
    """Show data analysis and visualizations"""
    st.header("📊 Data Analysis")
    
    st.info("This section would show data analysis from the preprocessing module. "
            "For now, this is a placeholder for future data visualization features.")
    
    # Placeholder for data analysis
    st.subheader("📈 Sample Data Visualizations")
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Sample fraud data
    fraud_data = pd.DataFrame({
        'Time': np.random.uniform(0, 172792, n_samples),
        'Amount': np.random.exponential(100, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    })
    
    # Time distribution
    fig1 = px.histogram(
        fraud_data,
        x='Time',
        color='Class',
        title='Transaction Time Distribution',
        labels={'Class': 'Transaction Type'},
        color_discrete_map={0: 'green', 1: 'red'}
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Amount distribution
    fig2 = px.box(
        fraud_data,
        x='Class',
        y='Amount',
        title='Transaction Amount Distribution',
        labels={'Class': 'Transaction Type'}
    )
    st.plotly_chart(fig2, use_container_width=True)

def show_api_status(health_data):
    """Show API status and information"""
    st.header("🔧 API Status")
    
    if health_data:
        st.success("✅ API is running and healthy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 System Information")
            st.json(health_data)
        
        with col2:
            st.subheader("🔗 API Endpoints")
            endpoints = [
                "GET / - Root endpoint",
                "GET /health - Health check",
                "POST /predict - Single prediction",
                "POST /predict/batch - Batch prediction",
                "GET /models - Model information",
                "POST /models/train - Train models",
                "GET /models/performance - Model performance",
                "GET /stats - Fraud statistics"
            ]
            
            for endpoint in endpoints:
                st.code(endpoint)
    else:
        st.error("❌ API is not responding")

if __name__ == "__main__":
    main()
