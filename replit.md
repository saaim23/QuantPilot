# Quantitative Finance Platform

## Overview

This is a comprehensive quantitative finance platform built with Streamlit that provides advanced options pricing, risk analysis, and portfolio management capabilities. The platform integrates multiple financial models including Black-Scholes, Heston, GARCH volatility models, and Monte Carlo simulations. It features machine learning-based option pricing, alternative data integration (satellite imagery, sentiment analysis), and sophisticated visualization tools for volatility surfaces and financial charts.

**Current Status**: Production-ready v1.0.0 with complete GitHub repository structure including comprehensive documentation, research papers, deployment guides, and professional licensing. The TensorFlow compatibility issue has been resolved by implementing Physics-Informed Neural Networks using scikit-learn while maintaining theoretical rigor.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with custom CSS styling
- **Layout**: Wide layout with expandable sidebar for navigation
- **Visualization**: Plotly for interactive charts, 3D volatility surfaces, and financial data visualization
- **User Interface**: Professional financial styling with responsive design

### Backend Architecture
- **Core Models**: Modular architecture with separate model classes for different financial instruments
  - Black-Scholes and Heston models for option pricing
  - GARCH and volatility modeling for risk analysis
  - Monte Carlo engines for simulation-based pricing
- **Machine Learning**: TensorFlow/Keras for neural networks, scikit-learn for traditional ML models
- **Calculation Engine**: Centralized financial calculations utility for Greeks, returns, and risk metrics

### Data Architecture
- **Market Data**: Multi-source data provider using Yahoo Finance and Alpha Vantage APIs
- **Alternative Data**: Satellite imagery analysis, ESG data, and retail footfall analytics
- **Sentiment Analysis**: TextBlob-based sentiment analysis for news and social media data
- **Real-time Processing**: Streaming data capabilities for live market updates

### Model Architecture
- **Option Pricing**: Multiple pricing models with configurable parameters
- **Risk Management**: VaR calculation, portfolio backtesting, and risk metrics
- **Feature Engineering**: Comprehensive feature creation for ML models including technical indicators
- **Backtesting Framework**: Strategy implementation and historical performance analysis

### Visualization Architecture
- **Chart Library**: Plotly-based interactive financial charts including candlestick, volume, and technical indicators
- **3D Surfaces**: Advanced volatility surface visualization with interpolation
- **Dashboard Layout**: Multi-panel dashboard with real-time updates

## External Dependencies

### APIs and Data Providers
- **Yahoo Finance**: Primary source for stock prices and options chain data via yfinance library
- **Alpha Vantage**: Alternative market data source for enhanced coverage
- **Twitter API**: Social media sentiment analysis (requires bearer token)
- **Satellite Data APIs**: Multiple providers for geospatial analytics (Planet, Orbital Insight, SpaceKnow)

### Python Libraries
- **Core Scientific**: NumPy, Pandas, SciPy for mathematical computations
- **Financial Modeling**: arch for GARCH models, yfinance for market data
- **Machine Learning**: scikit-learn, XGBoost, TensorFlow for ML models
- **Visualization**: Plotly for interactive charts and 3D surfaces
- **Web Framework**: Streamlit for the web application interface
- **Text Processing**: TextBlob for sentiment analysis, trafilatura for web scraping

### Configuration Management
- **Environment Variables**: API keys and sensitive configuration stored in environment variables
- **Settings Module**: Centralized configuration for financial parameters, data sources, and model defaults
- **Risk Parameters**: Configurable risk-free rates, simulation parameters, and trading constants

### Data Storage
- **In-Memory**: Pandas DataFrames for data manipulation and caching
- **Model Persistence**: joblib for saving trained ML models
- **Configuration**: JSON-based configuration for model parameters and settings