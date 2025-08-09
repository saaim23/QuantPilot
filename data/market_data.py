import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import streamlit as st
from typing import Dict, List, Optional, Tuple
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from config.settings import ALPHA_VANTAGE_API_KEY
import time

class MarketDataProvider:
    """Market data provider using multiple sources"""
    
    def __init__(self):
        self.av_ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    def get_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Get stock price data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Calculate additional metrics
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
            
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Dict:
        """Get options chain data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                st.error(f"No options data available for {symbol}")
                return {}
            
            # Use first available expiration if none specified
            if expiration_date is None:
                expiration_date = expirations[0]
            elif expiration_date not in expirations:
                st.warning(f"Expiration {expiration_date} not available. Using {expirations[0]}")
                expiration_date = expirations[0]
            
            # Get options chain for specific expiration
            options_chain = ticker.option_chain(expiration_date)
            
            return {
                'calls': options_chain.calls,
                'puts': options_chain.puts,
                'expiration_date': expiration_date,
                'available_expirations': expirations
            }
        except Exception as e:
            st.error(f"Error fetching options chain for {symbol}: {str(e)}")
            return {}
    
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-year treasury)"""
        try:
            # Try to get from FRED (Federal Reserve Economic Data)
            treasury = yf.Ticker("^TNX")
            data = treasury.history(period="5d")
            
            if not data.empty:
                rate = data['Close'].iloc[-1] / 100  # Convert percentage to decimal
                return rate
            else:
                # Fallback to default rate
                return 0.045  # 4.5% default
        except Exception:
            return 0.045  # 4.5% default
    
    def get_dividend_yield(self, symbol: str) -> float:
        """Get dividend yield for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            dividend_yield = info.get('dividendYield', 0)
            return dividend_yield if dividend_yield else 0
        except Exception:
            return 0
    
    def get_historical_volatility(self, symbol: str, window: int = 30) -> pd.Series:
        """Calculate historical volatility"""
        try:
            data = self.get_stock_data(symbol, period="1y")
            if data.empty:
                return pd.Series()
            
            returns = data['Returns'].dropna()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)
            return volatility
        except Exception as e:
            st.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return pd.Series()
    
    def get_vix_data(self) -> pd.DataFrame:
        """Get VIX (volatility index) data"""
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(period="2y")
            return data
        except Exception as e:
            st.error(f"Error fetching VIX data: {str(e)}")
            return pd.DataFrame()
    
    def get_crypto_data(self, symbol: str = "BTC-USD", period: str = "1y") -> pd.DataFrame:
        """Get cryptocurrency data"""
        try:
            crypto = yf.Ticker(symbol)
            data = crypto.history(period=period)
            
            if data.empty:
                st.error(f"No crypto data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Calculate additional metrics
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(365)  # Crypto trades 365 days
            
            return data
        except Exception as e:
            st.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_commodity_data(self, symbol: str = "GC=F", period: str = "1y") -> pd.DataFrame:
        """Get commodity data (Gold, Oil, etc.)"""
        try:
            commodity = yf.Ticker(symbol)
            data = commodity.history(period=period)
            
            if data.empty:
                st.error(f"No commodity data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Calculate additional metrics
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
            
            return data
        except Exception as e:
            st.error(f"Error fetching commodity data for {symbol}: {str(e)}")
            return pd.DataFrame()

class RealTimeDataProvider:
    """Real-time market data provider"""
    
    def __init__(self):
        self.alpha_vantage_key = ALPHA_VANTAGE_API_KEY
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time stock quote"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'beta': info.get('beta', 1),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            st.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            return {}
    
    def get_intraday_data(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Get intraday data using Alpha Vantage"""
        try:
            if self.alpha_vantage_key == "demo":
                st.warning("Using demo API key. Real-time data may be limited.")
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Error Message' in data:
                st.error(f"API Error: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data:
                st.warning(f"API Limit: {data['Note']}")
                return pd.DataFrame()
            
            # Parse time series data
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                st.error("Invalid response format from Alpha Vantage")
                return pd.DataFrame()
            
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            
            # Calculate additional metrics
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()

class MarketIndicators:
    """Market indicators and technical analysis"""
    
    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame, windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate moving averages"""
        result = data.copy()
        
        for window in windows:
            result[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
        
        return result
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        result = data.copy()
        
        # Calculate moving average and standard deviation
        ma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        
        result['BB_Upper'] = ma + (std * num_std)
        result['BB_Lower'] = ma - (std * num_std)
        result['BB_Middle'] = ma
        result['BB_Width'] = result['BB_Upper'] - result['BB_Lower']
        
        return result
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        result = data.copy()
        
        # Calculate price changes
        delta = data['Close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        result['RSI'] = 100 - (100 / (1 + rs))
        
        return result
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD"""
        result = data.copy()
        
        # Calculate EMAs
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        
        # Calculate MACD line
        result['MACD'] = ema_fast - ema_slow
        
        # Calculate signal line
        result['MACD_Signal'] = result['MACD'].ewm(span=signal).mean()
        
        # Calculate histogram
        result['MACD_Histogram'] = result['MACD'] - result['MACD_Signal']
        
        return result
    
    @staticmethod
    def calculate_volatility_metrics(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various volatility metrics"""
        returns = data['Returns'].dropna()
        
        return {
            'realized_volatility': returns.std() * np.sqrt(252),
            'volatility_30d': returns.tail(30).std() * np.sqrt(252),
            'volatility_60d': returns.tail(60).std() * np.sqrt(252),
            'volatility_90d': returns.tail(90).std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_drawdown': ((data['Close'] / data['Close'].cummax()) - 1).min(),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
        }

class SyntheticDataGenerator:
    """Generate synthetic market data for testing"""
    
    @staticmethod
    def generate_gbm_prices(S0: float, mu: float, sigma: float, T: float, 
                          n_steps: int = 252, n_paths: int = 1) -> pd.DataFrame:
        """Generate stock prices using Geometric Brownian Motion"""
        dt = T / n_steps
        
        # Generate random shocks
        Z = np.random.standard_normal((n_paths, n_steps))
        
        # Calculate price paths
        prices = []
        dates = pd.date_range(start=datetime.now(), periods=n_steps+1, freq='D')
        
        for path in range(n_paths):
            price_path = [S0]
            current_price = S0
            
            for i in range(n_steps):
                current_price *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[path, i])
                price_path.append(current_price)
            
            prices.append(price_path)
        
        # Create DataFrame
        if n_paths == 1:
            df = pd.DataFrame({
                'Date': dates,
                'Close': prices[0],
                'Open': np.array(prices[0]) * (1 + np.random.normal(0, 0.001, len(prices[0]))),
                'High': np.array(prices[0]) * (1 + np.abs(np.random.normal(0, 0.01, len(prices[0])))),
                'Low': np.array(prices[0]) * (1 - np.abs(np.random.normal(0, 0.01, len(prices[0])))),
                'Volume': np.random.randint(1000000, 10000000, len(prices[0]))
            })
            df.set_index('Date', inplace=True)
            
            # Calculate returns and volatility
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
        else:
            # Multiple paths
            df = pd.DataFrame(prices).T
            df.index = dates
            df.columns = [f'Path_{i+1}' for i in range(n_paths)]
        
        return df
    
    @staticmethod
    def generate_option_market_data(S0: float, strikes: List[float], expiries: List[float],
                                  r: float, base_vol: float = 0.2, n_observations: int = 1000) -> pd.DataFrame:
        """Generate synthetic option market data for ML training"""
        from models.black_scholes import BlackScholesModel
        
        market_data = []
        
        for _ in range(n_observations):
            # Random market conditions
            S = S0 * np.random.uniform(0.8, 1.2)  # Stock price variation
            vol_multiplier = np.random.uniform(0.5, 2.0)  # Volatility variation
            
            for K in strikes:
                for T in expiries:
                    # Add volatility smile/skew
                    moneyness = K / S
                    if moneyness < 0.95:  # OTM puts, ITM calls
                        vol = base_vol * vol_multiplier * (1 + 0.05 * (0.95 - moneyness))
                    elif moneyness > 1.05:  # ITM puts, OTM calls
                        vol = base_vol * vol_multiplier * (1 + 0.03 * (moneyness - 1.05))
                    else:  # ATM
                        vol = base_vol * vol_multiplier
                    
                    # Calculate theoretical prices
                    call_price = BlackScholesModel.option_price(S, K, T, r, vol, 'call')
                    put_price = BlackScholesModel.option_price(S, K, T, r, vol, 'put')
                    
                    # Add market noise
                    call_price *= np.random.uniform(0.95, 1.05)
                    put_price *= np.random.uniform(0.95, 1.05)
                    
                    # Call option data
                    market_data.append({
                        'spot_price': S,
                        'strike_price': K,
                        'time_to_expiry': T,
                        'risk_free_rate': r,
                        'option_type': 'call',
                        'option_price': call_price,
                        'implied_volatility': vol,
                        'moneyness': S / K,
                        'is_itm': 1 if S > K else 0
                    })
                    
                    # Put option data
                    market_data.append({
                        'spot_price': S,
                        'strike_price': K,
                        'time_to_expiry': T,
                        'risk_free_rate': r,
                        'option_type': 'put',
                        'option_price': put_price,
                        'implied_volatility': vol,
                        'moneyness': S / K,
                        'is_itm': 1 if S < K else 0
                    })
        
        return pd.DataFrame(market_data)
