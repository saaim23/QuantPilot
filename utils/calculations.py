import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Tuple
import math

class FinancialCalculations:
    """Core financial calculation utilities"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate log returns from price series"""
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate historical volatility"""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # Annualize assuming 252 trading days
        return vol
    
    @staticmethod
    def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def greeks_calculation(S: float, K: float, T: float, r: float, sigma: float, 
                          option_type: str = 'call') -> dict:
        """Calculate option Greeks"""
        d1, d2 = FinancialCalculations.d1_d2(S, K, T, r, sigma)
        
        # Common calculations
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        if option_type.lower() == 'call':
            delta = N_d1
            theta = -(S * n_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
        else:  # put
            delta = N_d1 - 1
            theta = -(S * n_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * (1 - N_d2)
        
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        vega = S * n_d1 * np.sqrt(T) / 100  # Per 1% change in volatility
        rho = K * T * np.exp(-r * T) * (N_d2 if option_type.lower() == 'call' else -(1 - N_d2)) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def implied_volatility_newton(market_price: float, S: float, K: float, T: float, 
                                r: float, option_type: str = 'call', 
                                max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        from models.black_scholes import BlackScholesModel
        
        # Initial guess
        sigma = 0.3
        
        for i in range(max_iterations):
            # Calculate option price and vega
            bs_price = BlackScholesModel.option_price(S, K, T, r, sigma, option_type)
            vega = FinancialCalculations.greeks_calculation(S, K, T, r, sigma, option_type)['vega'] * 100
            
            # Newton-Raphson update
            price_diff = bs_price - market_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            if vega == 0:
                break
                
            sigma = sigma - price_diff / vega
            
            # Ensure volatility stays positive
            sigma = max(sigma, 0.001)
            
        return sigma

class RiskMetrics:
    """Risk calculation utilities"""
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = RiskMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def maximum_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0
