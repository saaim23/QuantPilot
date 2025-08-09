import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class MonteCarloEngine:
    """Monte Carlo simulation engine for option pricing and risk analysis"""
    
    def __init__(self, n_simulations: int = 10000, n_steps: int = 252, 
                 random_seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        if random_seed:
            np.random.seed(random_seed)
    
    def geometric_brownian_motion(self, S0: float, mu: float, sigma: float, 
                                 T: float) -> np.ndarray:
        """Simulate stock prices using Geometric Brownian Motion"""
        dt = T / self.n_steps
        
        # Generate random shocks
        Z = np.random.standard_normal((self.n_simulations, self.n_steps))
        
        # Calculate log returns
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Calculate cumulative log returns
        log_S = np.log(S0) + np.cumsum(log_returns, axis=1)
        
        # Convert to prices
        S = np.exp(log_S)
        
        # Add initial price
        S_paths = np.column_stack([np.full(self.n_simulations, S0), S])
        
        return S_paths
    
    def heston_simulation(self, S0: float, v0: float, r: float, kappa: float,
                         theta: float, sigma: float, rho: float, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate stock prices and volatility using Heston model"""
        dt = T / self.n_steps
        
        # Initialize arrays
        S = np.zeros((self.n_simulations, self.n_steps + 1))
        v = np.zeros((self.n_simulations, self.n_steps + 1))
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Generate correlated random numbers
        for i in range(1, self.n_steps + 1):
            Z1 = np.random.standard_normal(self.n_simulations)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(self.n_simulations)
            
            # Update volatility (ensure it stays positive)
            v[:, i] = np.maximum(
                v[:, i-1] + kappa * (theta - v[:, i-1]) * dt + 
                sigma * np.sqrt(v[:, i-1]) * np.sqrt(dt) * Z2, 
                1e-6
            )
            
            # Update stock price
            S[:, i] = S[:, i-1] * np.exp(
                (r - 0.5 * v[:, i-1]) * dt + 
                np.sqrt(v[:, i-1]) * np.sqrt(dt) * Z1
            )
        
        return S, v
    
    def jump_diffusion_simulation(self, S0: float, mu: float, sigma: float, 
                                 lambda_j: float, mu_j: float, sigma_j: float, 
                                 T: float) -> np.ndarray:
        """Simulate stock prices with jump diffusion (Merton model)"""
        dt = T / self.n_steps
        
        # Initialize price paths
        S = np.zeros((self.n_simulations, self.n_steps + 1))
        S[:, 0] = S0
        
        for i in range(1, self.n_steps + 1):
            # Brownian motion component
            Z = np.random.standard_normal(self.n_simulations)
            
            # Jump component
            N_jumps = np.random.poisson(lambda_j * dt, self.n_simulations)
            jump_sizes = np.zeros(self.n_simulations)
            
            for j in range(self.n_simulations):
                if N_jumps[j] > 0:
                    jumps = np.random.normal(mu_j, sigma_j, N_jumps[j])
                    jump_sizes[j] = np.sum(jumps)
            
            # Update stock price
            S[:, i] = S[:, i-1] * np.exp(
                (mu - 0.5 * sigma**2 - lambda_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)) * dt +
                sigma * np.sqrt(dt) * Z + jump_sizes
            )
        
        return S
    
    def price_european_option(self, S_paths: np.ndarray, K: float, r: float, 
                             T: float, option_type: str = 'call') -> Dict:
        """Price European option using Monte Carlo"""
        # Final stock prices
        S_final = S_paths[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_final - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S_final, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate confidence interval
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        confidence_interval = (
            option_price - 1.96 * std_error * np.exp(-r * T),
            option_price + 1.96 * std_error * np.exp(-r * T)
        )
        
        return {
            'price': option_price,
            'std_error': std_error * np.exp(-r * T),
            'confidence_interval': confidence_interval,
            'payoffs': payoffs
        }
    
    def price_american_option_lsm(self, S_paths: np.ndarray, K: float, r: float, 
                                 T: float, option_type: str = 'put') -> Dict:
        """Price American option using Longstaff-Schwartz method"""
        dt = T / self.n_steps
        
        # Initialize cash flows matrix
        cash_flows = np.zeros((self.n_simulations, self.n_steps + 1))
        
        # Set terminal payoffs
        if option_type.lower() == 'call':
            cash_flows[:, -1] = np.maximum(S_paths[:, -1] - K, 0)
        else:
            cash_flows[:, -1] = np.maximum(K - S_paths[:, -1], 0)
        
        # Backward induction
        for t in range(self.n_steps - 1, 0, -1):
            # Current stock prices
            S_t = S_paths[:, t]
            
            # Intrinsic value
            if option_type.lower() == 'call':
                intrinsic = np.maximum(S_t - K, 0)
            else:
                intrinsic = np.maximum(K - S_t, 0)
            
            # Only consider in-the-money paths
            itm_mask = intrinsic > 0
            
            if np.sum(itm_mask) > 0:
                # Continuation value regression
                S_itm = S_t[itm_mask]
                future_cf = cash_flows[itm_mask, t+1:] * np.exp(-r * dt * np.arange(1, self.n_steps - t + 1))
                continuation_value = np.sum(future_cf, axis=1)
                
                # Regression: continuation value = a + b*S + c*S^2
                X = np.column_stack([np.ones(len(S_itm)), S_itm, S_itm**2])
                
                try:
                    coeffs = np.linalg.lstsq(X, continuation_value, rcond=None)[0]
                    estimated_continuation = X @ coeffs
                    
                    # Exercise decision
                    exercise_mask = intrinsic[itm_mask] > estimated_continuation
                    
                    # Update cash flows
                    cash_flows[itm_mask, t+1:] = 0  # Clear future cash flows
                    cash_flows[itm_mask, t][exercise_mask] = intrinsic[itm_mask][exercise_mask]
                except np.linalg.LinAlgError:
                    # If regression fails, use intrinsic value
                    cash_flows[itm_mask, t] = intrinsic[itm_mask]
        
        # Calculate option value
        discounted_cf = np.sum(cash_flows * np.exp(-r * dt * np.arange(self.n_steps + 1)), axis=1)
        option_price = np.mean(discounted_cf)
        
        return {
            'price': option_price,
            'std_error': np.std(discounted_cf) / np.sqrt(self.n_simulations),
            'cash_flows': cash_flows
        }
    
    def calculate_greeks_fd(self, S0: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> Dict:
        """Calculate Greeks using finite differences in Monte Carlo"""
        
        # Small perturbations
        dS = S0 * 0.01
        dr = 0.0001
        dsigma = 0.01
        dT = T * 0.01
        
        # Base case
        S_base = self.geometric_brownian_motion(S0, r, sigma, T)
        price_base = self.price_european_option(S_base, K, r, T, option_type)['price']
        
        # Delta: ∂V/∂S
        S_up = self.geometric_brownian_motion(S0 + dS, r, sigma, T)
        S_down = self.geometric_brownian_motion(S0 - dS, r, sigma, T)
        price_up = self.price_european_option(S_up, K, r, T, option_type)['price']
        price_down = self.price_european_option(S_down, K, r, T, option_type)['price']
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma: ∂²V/∂S²
        gamma = (price_up - 2 * price_base + price_down) / (dS ** 2)
        
        # Vega: ∂V/∂σ
        S_vega_up = self.geometric_brownian_motion(S0, r, sigma + dsigma, T)
        S_vega_down = self.geometric_brownian_motion(S0, r, sigma - dsigma, T)
        price_vega_up = self.price_european_option(S_vega_up, K, r, T, option_type)['price']
        price_vega_down = self.price_european_option(S_vega_down, K, r, T, option_type)['price']
        vega = (price_vega_up - price_vega_down) / (2 * dsigma)
        
        # Theta: -∂V/∂T
        S_theta = self.geometric_brownian_motion(S0, r, sigma, T - dT)
        price_theta = self.price_european_option(S_theta, K, r, T - dT, option_type)['price']
        theta = -(price_theta - price_base) / dT
        
        # Rho: ∂V/∂r
        S_rho_up = self.geometric_brownian_motion(S0, r + dr, sigma, T)
        S_rho_down = self.geometric_brownian_motion(S0, r - dr, sigma, T)
        price_rho_up = self.price_european_option(S_rho_up, K, r + dr, T, option_type)['price']
        price_rho_down = self.price_european_option(S_rho_down, K, r - dr, T, option_type)['price']
        rho = (price_rho_up - price_rho_down) / (2 * dr)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class VaRCalculator:
    """Value at Risk calculations using Monte Carlo"""
    
    @staticmethod
    def portfolio_var(returns: np.ndarray, confidence_level: float = 0.05) -> Dict:
        """Calculate portfolio VaR using historical simulation"""
        sorted_returns = np.sort(returns)
        var_index = int(confidence_level * len(returns))
        
        var = sorted_returns[var_index]
        cvar = np.mean(sorted_returns[:var_index])
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'worst_case': sorted_returns[0],
            'best_case': sorted_returns[-1]
        }
    
    @staticmethod
    def monte_carlo_var(portfolio_value: float, returns_sim: np.ndarray, 
                       confidence_level: float = 0.05) -> Dict:
        """Calculate VaR using Monte Carlo simulated returns"""
        portfolio_returns = returns_sim * portfolio_value
        return VaRCalculator.portfolio_var(portfolio_returns, confidence_level)

class PathDependentOptions:
    """Monte Carlo pricing for path-dependent options"""
    
    @staticmethod
    def asian_option(S_paths: np.ndarray, K: float, r: float, T: float,
                    option_type: str = 'call', average_type: str = 'arithmetic') -> Dict:
        """Price Asian option with geometric or arithmetic average"""
        
        if average_type == 'arithmetic':
            avg_prices = np.mean(S_paths, axis=1)
        elif average_type == 'geometric':
            avg_prices = np.exp(np.mean(np.log(S_paths), axis=1))
        else:
            raise ValueError("average_type must be 'arithmetic' or 'geometric'")
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return {
            'price': option_price,
            'std_error': np.std(payoffs) / np.sqrt(len(payoffs)) * np.exp(-r * T),
            'average_prices': avg_prices,
            'payoffs': payoffs
        }
    
    @staticmethod
    def barrier_option(S_paths: np.ndarray, K: float, B: float, r: float, T: float,
                      option_type: str = 'call', barrier_type: str = 'up_and_out') -> Dict:
        """Price barrier option"""
        
        # Check barrier condition for each path
        if barrier_type == 'up_and_out':
            barrier_hit = np.any(S_paths >= B, axis=1)
            active_paths = ~barrier_hit
        elif barrier_type == 'down_and_out':
            barrier_hit = np.any(S_paths <= B, axis=1)
            active_paths = ~barrier_hit
        elif barrier_type == 'up_and_in':
            barrier_hit = np.any(S_paths >= B, axis=1)
            active_paths = barrier_hit
        elif barrier_type == 'down_and_in':
            barrier_hit = np.any(S_paths <= B, axis=1)
            active_paths = barrier_hit
        else:
            raise ValueError("Invalid barrier_type")
        
        # Final stock prices
        S_final = S_paths[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.where(active_paths, np.maximum(S_final - K, 0), 0)
        else:
            payoffs = np.where(active_paths, np.maximum(K - S_final, 0), 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return {
            'price': option_price,
            'std_error': np.std(payoffs) / np.sqrt(len(payoffs)) * np.exp(-r * T),
            'barrier_hit_ratio': np.mean(barrier_hit),
            'payoffs': payoffs
        }
