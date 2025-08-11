# API Documentation

## Overview

The Quantitative Finance Platform provides a comprehensive API for pricing derivatives, managing risk, and implementing advanced AI-enhanced trading strategies. This document details all available modules, classes, and functions.

## Core Modules

### Option Pricing Models

#### BlackScholesModel

Classic Black-Scholes option pricing with Greeks calculation.

```python
from models.black_scholes import BlackScholesModel

# Price European option
price = BlackScholesModel.option_price(
    S=100,      # Current stock price
    K=105,      # Strike price
    T=0.25,     # Time to expiration (years)
    r=0.05,     # Risk-free rate
    sigma=0.20, # Volatility
    option_type='call'  # 'call' or 'put'
)

# Calculate Greeks
greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, option_type)
# Returns: {'delta': float, 'gamma': float, 'theta': float, 'vega': float, 'rho': float}
```

#### HestonModel

Advanced stochastic volatility modeling.

```python
from models.black_scholes import HestonModel

price = HestonModel.option_price_fft(
    S=100,       # Current stock price
    K=105,       # Strike price
    T=0.25,      # Time to expiration
    r=0.05,      # Risk-free rate
    v0=0.04,     # Initial volatility
    kappa=2.0,   # Mean reversion speed
    theta=0.04,  # Long-term volatility
    sigma_v=0.3, # Volatility of volatility
    rho=-0.7,    # Correlation between price and volatility
    option_type='call'
)
```

### Exotic Options

#### ExoticOptionsEngine

Advanced derivatives pricing for complex option structures.

```python
from models.exotic_options import ExoticOptionsEngine

engine = ExoticOptionsEngine()

# Barrier Options
barrier_result = engine.barrier_option_price(
    S=100,           # Current price
    K=105,           # Strike price
    T=0.25,          # Time to expiration
    r=0.05,          # Risk-free rate
    sigma=0.20,      # Volatility
    barrier=110,     # Barrier level
    option_type='call',
    barrier_type='knock_out'
)
# Returns: {'price': float, 'greeks': dict, 'probability_touch': float}

# Asian Options
asian_result = engine.asian_option_price(
    S=100, K=105, T=0.25, r=0.05, sigma=0.20,
    num_observations=252,  # Number of averaging observations
    option_type='call',
    avg_type='arithmetic'  # 'arithmetic' or 'geometric'
)

# Rainbow Options (Multi-Asset)
rainbow_result = engine.rainbow_option_price(
    S1=100, S2=95,        # Asset prices
    K=100,                # Strike price
    T=0.25, r=0.05,
    sigma1=0.20, sigma2=0.25,  # Individual volatilities
    correlation=0.3,      # Asset correlation
    option_type='max'     # 'max', 'min', or 'spread'
)
```

#### StructuredProducts

Complex financial instruments and structured notes.

```python
from models.exotic_options import StructuredProducts

structured = StructuredProducts()

# Autocallable Note
note_result = structured.autocallable_note(
    S=100,                    # Current stock price
    notional=1000,           # Notional amount
    coupon_rate=0.08,        # Annual coupon rate
    barrier_level=85,        # Knock-in barrier
    observation_dates=[0.25, 0.5, 0.75, 1.0],  # Quarterly observations
    protection_level=0.7     # Capital protection level
)
# Returns: {'value': float, 'coupon_pv': float, 'redemption_probability': float}
```

### Cryptocurrency Derivatives

#### CryptoDerivativesEngine

Specialized pricing for cryptocurrency derivatives.

```python
from models.crypto_derivatives import CryptoDerivativesEngine

crypto_engine = CryptoDerivativesEngine()

# Crypto Options with Market Adjustments
crypto_option = crypto_engine.crypto_option_price(
    S=50000,              # Bitcoin price
    K=52000,              # Strike price
    T=0.25,               # Time to expiration
    r=0.05,               # Risk-free rate
    sigma=0.80,           # Crypto volatility
    option_type='call',
    crypto_adjustments=True  # Apply 24/7 trading and jump risk adjustments
)
# Returns: {'price': float, 'adjusted_volatility': float, 'jump_component': float}

# Perpetual Futures
perp_result = crypto_engine.perpetual_futures_price(
    S=50000,                 # Spot price
    funding_rate=0.0001,     # 8-hour funding rate
    premium_index=0.0        # Premium/discount to spot
)
# Returns: {'mark_price': float, 'basis': float, 'funding_payment': float}

# DeFi Options
defi_option = crypto_engine.defi_option_price(
    S=100, K=105, T=0.25,
    protocol_risk=0.05,      # Smart contract risk premium
    liquidity_premium=0.02,  # Liquidity risk premium
    gas_cost=50             # Gas cost in USD
)
```

### AI-Enhanced Models

#### QuantumInspiredOptimizer

Next-generation portfolio optimization using quantum-inspired algorithms.

```python
from models.ai_enhanced_models import QuantumInspiredOptimizer

optimizer = QuantumInspiredOptimizer()

# Quantum Portfolio Optimization
result = optimizer.quantum_portfolio_optimization(
    returns=returns_dataframe,  # DataFrame of asset returns
    risk_tolerance=0.5,         # 0 = conservative, 1 = aggressive
    quantum_iterations=1000     # Number of quantum optimization cycles
)
# Returns: {'optimal_weights': array, 'expected_return': float, 'sharpe_ratio': float}
```

#### ReinforcementLearningTrader

Autonomous trading agent using deep reinforcement learning.

```python
from models.ai_enhanced_models import ReinforcementLearningTrader

rl_trader = ReinforcementLearningTrader()

# Train Trading Agent
training_results = rl_trader.train_rl_agent(
    price_data=stock_data,      # OHLCV DataFrame
    lookback_window=20,         # Feature window size
    episodes=1000              # Training episodes
)
# Returns: {'average_reward': float, 'win_rate': float, 'q_table_size': int}
```

#### TransformerPricePredictor

Advanced neural network with attention mechanisms for price prediction.

```python
from models.ai_enhanced_models import TransformerPricePredictor

transformer = TransformerPricePredictor()

# Create Feature Sequences
sequences = transformer.create_transformer_features(price_data)

# Run Attention Analysis
attention_results = transformer.attention_mechanism_simulation(sequences)
# Returns: {'predictions': list, 'attention_weights': array, 'feature_importance': list}
```

### Real-Time Risk Management

#### RealTimeRiskEngine

Professional-grade risk monitoring and management system.

```python
from models.real_time_risk_engine import RealTimeRiskEngine

risk_engine = RealTimeRiskEngine()

# Real-Time Risk Monitoring
risk_results = risk_engine.real_time_risk_monitor(
    portfolio={'weights': [0.4, 0.6], 'values': [40000, 60000]},
    market_data=price_dataframe,
    confidence_level=0.95
)
# Returns comprehensive risk metrics and alerts

# Dynamic Position Sizing
sizing_result = risk_engine.dynamic_position_sizing(
    symbol='AAPL',
    current_volatility=0.25,    # Annualized volatility
    target_risk=0.02,           # 2% risk target
    portfolio_value=100000
)
# Returns: {'optimal_position_size': float, 'kelly_fraction': float}
```

### Volatility Models

#### GARCHModel

Time-series volatility forecasting using GARCH models.

```python
from models.volatility import GARCHModel

garch = GARCHModel()

# Fit GARCH Model
garch_results = garch.fit_garch(returns_series, p=1, q=1)

# Forecast Volatility
volatility_forecast = garch.forecast_volatility(
    fitted_model=garch_results['model'],
    horizon=10  # 10-day forecast
)
```

#### ImpliedVolatilitySurface

3D implied volatility surface construction and analysis.

```python
from models.volatility import ImpliedVolatilitySurface

iv_surface = ImpliedVolatilitySurface()

# Build Volatility Surface
surface_data = iv_surface.build_surface(
    option_chain_data,     # Options market data
    risk_free_rate=0.05,
    dividend_yield=0.02
)

# Interpolate Volatility
interpolated_vol = iv_surface.interpolate_volatility(
    strike=105, 
    time_to_expiry=0.25,
    surface_data=surface_data
)
```

### Monte Carlo Simulations

#### MonteCarloEngine

High-performance Monte Carlo pricing engine.

```python
from models.monte_carlo import MonteCarloEngine

mc_engine = MonteCarloEngine(n_simulations=100000)

# Generate Price Paths
price_paths = mc_engine.geometric_brownian_motion(
    S0=100,      # Initial price
    r=0.05,      # Risk-free rate
    sigma=0.20,  # Volatility
    T=1.0        # Time horizon
)

# Price European Option
option_result = mc_engine.price_european_option(
    price_paths, K=105, r=0.05, T=1.0, option_type='call'
)

# Calculate VaR
var_result = mc_engine.calculate_var(
    portfolio_returns,
    confidence_level=0.95,
    time_horizon=1
)
```

### Market Data Providers

#### MarketDataProvider

Real-time and historical market data integration.

```python
from data.market_data import MarketDataProvider

data_provider = MarketDataProvider()

# Get Stock Data
stock_data = data_provider.get_stock_data(
    symbol='AAPL',
    period='1y',      # '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    interval='1d'     # '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
)

# Get Option Chain
option_chain = data_provider.get_option_chain('AAPL', '2024-03-15')

# Get Real-Time Quote
quote = data_provider.get_real_time_quote('AAPL')
# Returns: {'symbol': str, 'price': float, 'change': float, 'volume': int}
```

#### AlternativeDataAggregator

Integration with satellite imagery, ESG, and alternative data sources.

```python
from data.alternative_data import AlternativeDataAggregator

alt_data = AlternativeDataAggregator()

# ESG Scores
esg_data = alt_data.get_esg_scores(['AAPL', 'GOOGL', 'MSFT'])

# Satellite Economic Indicators
satellite_data = alt_data.get_satellite_indicators(
    companies=['AAPL', 'TSLA'],
    indicator_type='parking_lots'  # Economic activity proxy
)
```

### Backtesting Framework

#### OptionTradingStrategy

Comprehensive strategy backtesting with multiple option strategies.

```python
from backtesting.strategy import OptionTradingStrategy

strategy = OptionTradingStrategy()

# Backtest Covered Call Strategy
backtest_results = strategy.backtest_covered_call(
    stock_data=price_data,
    strike_selection='otm_5',    # 5% out-of-the-money
    expiration_days=30,          # 30-day expiration
    initial_capital=100000
)

# Backtest Iron Condor
condor_results = strategy.backtest_iron_condor(
    stock_data=price_data,
    width=10,                    # Strike width
    expiration_days=45,
    initial_capital=100000
)
```

### Risk Metrics and Calculations

#### RiskMetrics

Comprehensive risk calculation utilities.

```python
from utils.calculations import RiskMetrics

risk_metrics = RiskMetrics()

# Portfolio VaR
portfolio_var = risk_metrics.portfolio_var(
    weights=portfolio_weights,
    returns_matrix=returns_data,
    confidence_level=0.95
)

# Maximum Drawdown
max_dd = risk_metrics.maximum_drawdown(portfolio_returns)

# Sharpe Ratio
sharpe = risk_metrics.sharpe_ratio(
    returns=portfolio_returns,
    risk_free_rate=0.02
)

# Greeks Portfolio
portfolio_greeks = risk_metrics.portfolio_greeks(
    positions=option_positions,
    market_data=current_prices
)
```

## Error Handling

All API functions include comprehensive error handling with informative error messages:

```python
try:
    result = BlackScholesModel.option_price(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Calculation error: {e}")
```

## Performance Considerations

### Caching

The platform uses Streamlit's caching system for improved performance:

```python
@st.cache_resource
def get_market_data_provider():
    return MarketDataProvider()

@st.cache_data
def calculate_option_prices(S, K, T, r, sigma):
    return BlackScholesModel.option_price(S, K, T, r, sigma)
```

### Optimization Tips

1. **Batch Operations**: Use vectorized NumPy operations for multiple calculations
2. **Data Caching**: Cache market data to avoid repeated API calls
3. **Parameter Validation**: Validate inputs before expensive calculations
4. **Memory Management**: Use generators for large Monte Carlo simulations

## Configuration

### Environment Variables

```bash
# Optional API keys for enhanced functionality
ALPHA_VANTAGE_API_KEY=your_key_here
TWITTER_BEARER_TOKEN=your_token_here

# Risk management parameters
DEFAULT_RISK_FREE_RATE=0.05
MONTE_CARLO_SIMULATIONS=100000
MAX_VAR_CONFIDENCE=0.99
```

### Settings Configuration

```python
# config/settings.py
RISK_FREE_RATE = 0.05
MONTE_CARLO_SIMULATIONS = 100000
DEFAULT_VOLATILITY = 0.20
CACHE_TIMEOUT = 3600  # 1 hour
```

## Integration Examples

### Complete Option Pricing Workflow

```python
# 1. Get market data
data_provider = MarketDataProvider()
stock_data = data_provider.get_stock_data('AAPL', '1y')

# 2. Calculate implied volatility
current_price = stock_data['Close'].iloc[-1]
historical_vol = stock_data['Close'].pct_change().std() * np.sqrt(252)

# 3. Price options
bs_price = BlackScholesModel.option_price(
    S=current_price, K=current_price*1.05, T=0.25, 
    r=0.05, sigma=historical_vol, option_type='call'
)

# 4. Compare with exotic options
exotic_engine = ExoticOptionsEngine()
barrier_price = exotic_engine.barrier_option_price(
    S=current_price, K=current_price*1.05, T=0.25,
    r=0.05, sigma=historical_vol, barrier=current_price*1.10,
    option_type='call', barrier_type='knock_out'
)

# 5. Risk analysis
risk_engine = RealTimeRiskEngine()
portfolio = {'weights': [1.0], 'values': [bs_price]}
risk_metrics = risk_engine.real_time_risk_monitor(portfolio, stock_data)
```

### AI-Enhanced Trading Strategy

```python
# 1. Prepare data
transformer = TransformerPricePredictor()
features = transformer.create_transformer_features(stock_data)

# 2. Train RL agent
rl_trader = ReinforcementLearningTrader()
training_results = rl_trader.train_rl_agent(stock_data, episodes=1000)

# 3. Optimize portfolio
optimizer = QuantumInspiredOptimizer()
returns_df = stock_data[['AAPL', 'GOOGL', 'MSFT']].pct_change().dropna()
optimal_weights = optimizer.quantum_portfolio_optimization(returns_df)

# 4. Monitor risk
risk_results = risk_engine.real_time_risk_monitor(
    portfolio=optimal_weights, 
    market_data=stock_data
)
```

## Support and Documentation

- **GitHub Issues**: Report bugs and request features
- **API Reference**: Complete function documentation with examples
- **Examples**: Sample implementations in `/examples` directory
- **Research Papers**: Academic foundation in `/docs` directory

---

**Note**: This API is designed for educational and research purposes. For production trading, ensure proper risk management and regulatory compliance.