import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Dict, List
import streamlit as st

class BlackScholesModel:
    """Black-Scholes European option pricing model implementation"""
    
    @staticmethod
    def option_price(S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        """
        if T <= 0:
            # Option has expired
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return price
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str = 'call') -> Dict[str, float]:
        """Calculate all Greeks for an option"""
        from utils.calculations import FinancialCalculations
        return FinancialCalculations.greeks_calculation(S, K, T, r, sigma, option_type)
    
    @staticmethod
    def create_option_chain(S: float, strikes: List[float], T: float, r: float, 
                           sigma: float) -> pd.DataFrame:
        """Create option chain with calls and puts"""
        chain_data = []
        
        for K in strikes:
            call_price = BlackScholesModel.option_price(S, K, T, r, sigma, 'call')
            put_price = BlackScholesModel.option_price(S, K, T, r, sigma, 'put')
            
            call_greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, 'call')
            put_greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, 'put')
            
            chain_data.append({
                'Strike': K,
                'Call_Price': call_price,
                'Call_Delta': call_greeks['delta'],
                'Call_Gamma': call_greeks['gamma'],
                'Call_Theta': call_greeks['theta'],
                'Call_Vega': call_greeks['vega'],
                'Put_Price': put_price,
                'Put_Delta': put_greeks['delta'],
                'Put_Gamma': put_greeks['gamma'],
                'Put_Theta': put_greeks['theta'],
                'Put_Vega': put_greeks['vega'],
                'Intrinsic_Call': max(S - K, 0),
                'Intrinsic_Put': max(K - S, 0),
                'Time_Value_Call': call_price - max(S - K, 0),
                'Time_Value_Put': put_price - max(K - S, 0)
            })
        
        return pd.DataFrame(chain_data)
    
    @staticmethod
    def sensitivity_analysis(S: float, K: float, T: float, r: float, sigma: float,
                           option_type: str = 'call') -> Dict[str, pd.DataFrame]:
        """Perform sensitivity analysis across different parameters"""
        
        # Price sensitivity to underlying
        spot_range = np.linspace(S * 0.7, S * 1.3, 50)
        spot_prices = [BlackScholesModel.option_price(s, K, T, r, sigma, option_type) 
                      for s in spot_range]
        
        # Price sensitivity to volatility
        vol_range = np.linspace(0.1, 1.0, 50)
        vol_prices = [BlackScholesModel.option_price(S, K, T, r, v, option_type) 
                     for v in vol_range]
        
        # Price sensitivity to time
        time_range = np.linspace(0.01, T, 50)
        time_prices = [BlackScholesModel.option_price(S, K, t, r, sigma, option_type) 
                      for t in time_range]
        
        return {
            'spot_sensitivity': pd.DataFrame({
                'Spot_Price': spot_range,
                'Option_Price': spot_prices
            }),
            'volatility_sensitivity': pd.DataFrame({
                'Volatility': vol_range,
                'Option_Price': vol_prices
            }),
            'time_sensitivity': pd.DataFrame({
                'Time_to_Expiry': time_range,
                'Option_Price': time_prices
            })
        }

class HestonModel:
    """Heston stochastic volatility model implementation"""
    
    @staticmethod
    def characteristic_function(phi: complex, S0: float, v0: float, kappa: float, 
                              theta: float, sigma: float, rho: float, 
                              r: float, T: float) -> complex:
        """Heston characteristic function"""
        # Model parameters
        xi = kappa - rho * sigma * phi * 1j
        d = np.sqrt(xi**2 + sigma**2 * (phi * 1j + phi**2))
        
        A1 = phi * 1j * (np.log(S0) + r * T)
        A2 = (kappa * theta) / (sigma**2) * (xi - d) * T
        A3 = -(kappa * theta) / (sigma**2) * np.log((1 - (xi - d) * np.exp(-d * T) / (2 * d)) / 
                                                     (1 - (xi - d) / (2 * d)))
        A4 = -(v0 / sigma**2) * (xi - d) * (1 - np.exp(-d * T)) / (1 - (xi - d) * np.exp(-d * T) / (2 * d))
        
        return np.exp(A1 + A2 + A3 + A4)
    
    @staticmethod
    def option_price_fft(S0: float, K: float, T: float, r: float, v0: float,
                        kappa: float, theta: float, sigma: float, rho: float,
                        option_type: str = 'call') -> float:
        """Price options using FFT method (simplified implementation)"""
        # This is a simplified version - full implementation would use FFT
        # For demonstration, we'll use an approximation
        
        # Monte Carlo approximation for Heston
        n_paths = 10000
        dt = T / 252
        
        S = np.zeros((n_paths, int(T/dt) + 1))
        v = np.zeros((n_paths, int(T/dt) + 1))
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        for i in range(1, int(T/dt) + 1):
            Z1 = np.random.normal(0, 1, n_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_paths)
            
            v[:, i] = np.maximum(v[:, i-1] + kappa * (theta - v[:, i-1]) * dt + 
                                sigma * np.sqrt(v[:, i-1]) * np.sqrt(dt) * Z2, 0)
            
            S[:, i] = S[:, i-1] * np.exp((r - 0.5 * v[:, i-1]) * dt + 
                                        np.sqrt(v[:, i-1]) * np.sqrt(dt) * Z1)
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)
        
        return np.exp(-r * T) * np.mean(payoffs)
