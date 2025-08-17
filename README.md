# Cryptocurrency Price Prediction System

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![ML Pipeline](https://img.shields.io/badge/pipeline-feature%20engineering%20%7C%20model%20training%20%7C%20backtesting-orange.svg)

> Machine learning system for cryptocurrency trend prediction and quantitative trading
>
> [![Star this Repo](https://img.shields.io/github/stars/hellosummer12315/predicting-crypto?style=social)](https://github.com/hellosummer12315/predicting-crypto)

## Key Features

- **Feature Engineering**
  - Technical indicators (MACD/Bollinger Bands/Stochastic)
  - Hidden Markov Models for regime detection
  - Trend scanning label generation

- **Hybrid Model Architecture**
  - Random Forest (time-series cross-validated)
  - XGBoost (Bayesian-optimized)
  - LSTM Neural Networks (sequence pattern recognition)

- **Backtesting**
  - Three training strategies (rolling/expanding/one-time)
  - Backtrader integration (stop-loss/dynamic position sizing)
  - Performance metrics (Sharpe ratio/Max Drawdown)

## Quick Start

### Prerequisites
```bash
# Clone repository
git clone https://github.com/yourusername/crypto-prediction.git
cd crypto-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
