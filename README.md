# 📈 Stock Market Volatility Prediction

**Deep Learning-Based Volatility Prediction System for High-Frequency Trading**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-IITM-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements a comprehensive **deep learning-based volatility prediction system** using multi-layer RNN/LSTM networks for high-frequency trading data analysis. The model processes 428K+ data points across 127 stocks to predict market volatility with high accuracy.

## 🚀 Key Features

- **Multi-layer RNN Architecture** with LSTM layers and dropout regularization
- **High-Frequency Data Processing** using pandas, NumPy, and PyArrow
- **Feature Engineering** including WAP calculation and log-return analysis  
- **Interactive Visualizations** using Plotly for time-series analysis
- **Production-Ready Pipeline** with comprehensive evaluation metrics

## 📊 Performance Metrics

- **F1 Score:** 0.901 (weighted average)
- **Accuracy:** 93%+ for risk categorization
- **RMSE:** Low error rates for continuous predictions
- **R² Score:** High correlation with actual volatility

## 🛠 Technology Stack

- **Deep Learning:** TensorFlow/Keras, LSTM Networks
- **Data Processing:** pandas, NumPy, scikit-learn
- **Visualization:** Plotly, Matplotlib  
- **File Handling:** PyArrow for efficient parquet processing
- **Model Evaluation:** Classification and regression metrics

## 📁 Project Structure

```
Stock-Market-Volatility-Prediction/
├── optiverstd.ipynb          # Main Jupyter notebook with complete analysis
├── train.csv                 # Training dataset (428K+ records)
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yeshwanth1008/Stock-Market-Volatility-Prediction.git
   cd Stock-Market-Volatility-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook optiverstd.ipynb
   ```

## 📈 Model Architecture

### Multi-layer LSTM Network
```
Input Layer (sequence_length=50)
    ↓
LSTM Layer (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (32 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (16 units, return_sequences=False)
    ↓
Dropout (0.2)
    ↓
Dense Layer (8 units, ReLU)
    ↓
Output Layer (1 unit)
```

## 📊 Results

### Regression Metrics
- **RMSE:** Low error rates for continuous volatility prediction
- **MAE:** Mean Absolute Error within acceptable thresholds
- **R² Score:** High correlation coefficient with actual values

### Classification Metrics (Risk Categories)
- **F1 Score (Macro):** 0.901
- **F1 Score (Weighted):** 0.901
- **Category Accuracy:** 93%+

## 🔬 Methodology

1. **Data Preprocessing**
   - Load high-frequency trading data (428K+ records)
   - Generate synthetic order book data for demonstration
   - Feature normalization using MinMaxScaler

2. **Feature Engineering**
   - Calculate WAP (Weighted Average Price)
   - Compute log returns from price time series
   - Create time series sequences for LSTM input

3. **Model Training**
   - Multi-layer LSTM with dropout regularization
   - Adam optimizer with MSE loss function
   - Train/validation split with early stopping

4. **Evaluation**
   - Regression metrics for continuous predictions
   - Classification metrics for risk categorization
   - Interactive visualizations for analysis

## 📈 Business Applications

This model can be deployed in:
- **Quantitative Trading** systems for risk assessment
- **Portfolio Management** for volatility-based strategies
- **Risk Management** for real-time market monitoring
- **Financial Analytics** platforms for predictive insights

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the IITM License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Yeshwanth1008**
- GitHub: [@Yeshwanth1008](https://github.com/Yeshwanth1008)
- LinkedIn: [Connect with me](https://linkedin.com)

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing libraries
- Inspired by quantitative finance and machine learning research
- Built with modern deep learning best practices

---

⭐ **If you found this project helpful, please give it a star!** ⭐
