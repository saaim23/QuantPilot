# API Documentation

## Overview

This document provides comprehensive documentation for the Quantitative Finance Platform's internal APIs and functions. While the platform primarily runs as a Streamlit web application, the underlying models and functions can be used programmatically.

## Core Models API

### Black-Scholes Model

#### `BlackScholesModel.option_price(S, K, T, r, sigma, option_type)`

Calculate option price using Black-Scholes formula.

**Parameters:**
- `S` (float): Current stock price
- `K` (float): Strike price  
- `T` (float): Time to expiration (years)
- `r` (float): Risk-free rate (decimal)
- `sigma` (float): Volatility (decimal)
- `option_type` (str): 'call' or 'put'

**Returns:**
- `float`: Option price

**Example:**
```python
from models.black_scholes import BlackScholesModel

price = BlackScholesModel.option_price(
    S=100,      # Current price
    K=105,      # Strike price
    T=0.25,     # 3 months
    r=0.05,     # 5% risk-free rate
    sigma=0.20, # 20% volatility
    option_type='call'
)
print(f"Call option price: ${price:.2f}")
```

#### `BlackScholesModel.calculate_greeks(S, K, T, r, sigma, option_type)`

Calculate option Greeks (delta, gamma, theta, vega, rho).

**Returns:**
- `dict`: Dictionary containing all Greeks

**Example:**
```python
greeks = BlackScholesModel.calculate_greeks(100, 105, 0.25, 0.05, 0.20, 'call')
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
```

### Machine Learning Models

#### `MLOptionPricer.train_models(data, target_column, test_size)`

Train multiple ML models for option pricing.

**Parameters:**
- `data` (pd.DataFrame): Training data with features
- `target_column` (str): Name of target price column
- `test_size` (float): Test set proportion (0-1)

**Returns:**
- `dict`: Training results and metrics

**Example:**
```python
from models.ml_models import MLOptionPricer

pricer = MLOptionPricer()
results = pricer.train_models(
    data=option_data,
    target_column='option_price',
    test_size=0.2
)

# View model performance
for model_name, metrics in results.items():
    print(f"{model_name}: R² = {metrics['test_metrics']['r2']:.3f}")
```

#### `MLOptionPricer.predict(S, K, T, r, sigma, model_name)`

Predict option price using trained ML model.

**Example:**
```python
prediction = pricer.predict(
    S=100, K=105, T=0.25, r=0.05, sigma=0.20, 
    model_name='xgboost'
)
```

### Volatility Models

#### `GARCHModel.fit(returns, p, q)`

Fit GARCH(p,q) model to return series.

**Parameters:**
- `returns` (pd.Series): Return time series
- `p` (int): GARCH order
- `q` (int): ARCH order

**Example:**
```python
from models.volatility import GARCHModel

garch = GARCHModel()
model = garch.fit(returns, p=1, q=1)
volatility_forecast = garch.forecast(horizon=30)
```

### Monte Carlo Engine

#### `MonteCarloEngine.european_option(S, K, T, r, sigma, option_type, simulations)`

Price European options using Monte Carlo simulation.

**Example:**
```python
from models.monte_carlo import MonteCarloEngine

mc = MonteCarloEngine()
price, confidence_interval = mc.european_option(
    S=100, K=105, T=0.25, r=0.05, sigma=0.20,
    option_type='call', simulations=100000
)
```

## Data Providers API

### Market Data Provider

#### `MarketDataProvider.get_stock_data(symbol, period, interval)`

Fetch historical stock data.

**Example:**
```python
from data.market_data import MarketDataProvider

provider = MarketDataProvider()
data = provider.get_stock_data('AAPL', period='1y', interval='1d')
```

#### `MarketDataProvider.get_options_chain(symbol, expiration_date)`

Get options chain for specific expiration.

**Example:**
```python
options_chain = provider.get_options_chain('AAPL', '2024-03-15')
```

### Alternative Data

#### `SentimentAnalyzer.analyze_news(symbol, days_back)`

Analyze news sentiment for a stock.

**Example:**
```python
from data.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment = analyzer.analyze_news('AAPL', days_back=30)
print(f"Average sentiment: {sentiment['average_score']:.2f}")
```

## Visualization API

### Financial Charts

#### `FinancialCharts.candlestick_chart(data, title)`

Create candlestick chart.

**Example:**
```python
from visualization.charts import FinancialCharts

charts = FinancialCharts()
fig = charts.candlestick_chart(stock_data, "AAPL Price History")
fig.show()
```

### Volatility Surface

#### `VolatilitySurfaceVisualizer.create_3d_surface(surface_data)`

Create 3D implied volatility surface.

**Example:**
```python
from visualization.volatility_surface import VolatilitySurfaceVisualizer

visualizer = VolatilitySurfaceVisualizer()
fig = visualizer.create_3d_surface(iv_surface_data)
```

## Backtesting API

### Strategy Engine

#### `OptionTradingStrategy.long_call_strategy(S, K, T, r, sigma, position_size)`

Define long call strategy.

**Example:**
```python
from backtesting.strategy import OptionTradingStrategy

strategy = OptionTradingStrategy()
position = strategy.long_call_strategy(
    S=100, K=105, T=0.25, r=0.05, sigma=0.20, position_size=10
)
print(f"Max profit: ${position['max_profit']}")
```

#### `PortfolioBacktester.run_backtest(universe_data, strategy_config, start_date, end_date)`

Run comprehensive portfolio backtest.

**Example:**
```python
from backtesting.strategy import PortfolioBacktester

backtester = PortfolioBacktester()
results = backtester.run_backtest(
    universe_data={'AAPL': aapl_data},
    strategy_config={
        'initial_capital': 100000,
        'position_size_pct': 0.1,
        'strategy_type': 'covered_call'
    },
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

## Utility Functions API

### Financial Calculations

#### `FinancialCalculations.calculate_returns(prices, method)`

Calculate returns from price series.

**Parameters:**
- `method` (str): 'simple' or 'log'

**Example:**
```python
from utils.calculations import FinancialCalculations

returns = FinancialCalculations.calculate_returns(price_series, method='log')
```

#### `RiskMetrics.calculate_var(returns, confidence_level, method)`

Calculate Value at Risk.

**Example:**
```python
from utils.calculations import RiskMetrics

var_95 = RiskMetrics.calculate_var(returns, confidence_level=0.95, method='historical')
```

## Configuration API

### Settings

Access configuration parameters:

```python
from config.settings import RISK_FREE_RATE, MONTE_CARLO_SIMULATIONS

print(f"Default risk-free rate: {RISK_FREE_RATE}")
print(f"Default MC simulations: {MONTE_CARLO_SIMULATIONS}")
```

## Error Handling

All API functions include comprehensive error handling:

```python
try:
    price = BlackScholesModel.option_price(100, 105, 0.25, 0.05, 0.20, 'call')
except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"Calculation error: {e}")
```

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Use vectorized operations when possible
2. **Caching**: Results are cached for repeated calculations
3. **Memory Management**: Large datasets are processed in chunks
4. **Parallel Processing**: Monte Carlo simulations use multiprocessing

### Memory Usage

```python
import psutil
import gc

# Monitor memory usage
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Clean up after large calculations
gc.collect()
```

## Testing

### Unit Tests

Run tests for specific modules:

```bash
# Test option pricing models
python -m pytest tests/test_black_scholes.py -v

# Test machine learning models
python -m pytest tests/test_ml_models.py -v

# Test all modules
python -m pytest tests/ -v
```

### Performance Tests

```python
import time
from models.black_scholes import BlackScholesModel

# Benchmark option pricing
start_time = time.time()
for i in range(10000):
    price = BlackScholesModel.option_price(100, 105, 0.25, 0.05, 0.20, 'call')
end_time = time.time()

print(f"10,000 calculations took {end_time - start_time:.2f} seconds")
```

## Contributing to the API

When adding new API functions:

1. **Documentation**: Include comprehensive docstrings
2. **Type Hints**: Use proper type annotations
3. **Error Handling**: Implement robust error checking
4. **Tests**: Write unit tests for new functions
5. **Examples**: Provide usage examples

### Template for New Functions

```python
def new_function(param1: float, param2: str = 'default') -> dict:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 'default')
    
    Returns:
        Dictionary containing results
        
    Raises:
        ValueError: If parameters are invalid
        
    Example:
        >>> result = new_function(100.0, 'test')
        >>> print(result['value'])
    """
    # Implementation here
    pass
```