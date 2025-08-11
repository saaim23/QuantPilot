"""
Exotic Options Pricing Models
Advanced option types beyond vanilla calls and puts
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import streamlit as st
from typing import Dict, List, Tuple, Optional

class ExoticOptionsEngine:
    """Advanced exotic options pricing engine"""
    
    def __init__(self):
        self.name = "Exotic Options Engine"
    
    def barrier_option_price(self, S: float, K: float, T: float, r: float, 
                            sigma: float, barrier: float, option_type: str = 'call',
                            barrier_type: str = 'knock_out') -> Dict:
        """
        Price barrier options (knock-in/knock-out)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            barrier: Barrier level
            option_type: 'call' or 'put'
            barrier_type: 'knock_out', 'knock_in', 'up_and_out', 'down_and_out'
        """
        try:
            # Barrier option parameters
            mu = (r - 0.5 * sigma**2) / sigma**2
            lambda_val = np.sqrt(mu**2 + 2*r/sigma**2)
            
            if barrier_type in ['knock_out', 'up_and_out']:
                # Up-and-out barrier call
                if S >= barrier:
                    price = 0.0  # Already knocked out
                else:
                    # Standard Black-Scholes adjusted for barrier
                    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    
                    # Barrier adjustment terms
                    d1_barrier = (np.log(S/barrier) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2_barrier = d1_barrier - sigma*np.sqrt(T)
                    
                    vanilla_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
                    barrier_adjustment = (barrier/S)**(2*lambda_val) * (
                        barrier * norm.cdf(d1_barrier) - K * np.exp(-r*T) * norm.cdf(d2_barrier)
                    )
                    
                    price = vanilla_price - barrier_adjustment
                    
            elif barrier_type in ['knock_in', 'down_and_in']:
                # Knock-in option = Vanilla - Knock-out
                vanilla_call = S * norm.cdf((np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))) - K * np.exp(-r*T) * norm.cdf((np.log(S/K) + (r - 0.5*sigma**2)*T)/(sigma*np.sqrt(T)))
                knock_out_price = self.barrier_option_price(S, K, T, r, sigma, barrier, option_type, 'knock_out')['price']
                price = vanilla_call - knock_out_price
            
            # Calculate Greeks for barrier options
            greeks = self._calculate_barrier_greeks(S, K, T, r, sigma, barrier, price)
            
            return {
                'price': max(0, price),
                'greeks': greeks,
                'barrier_level': barrier,
                'type': f"{barrier_type}_{option_type}",
                'probability_touch': self._barrier_touch_probability(S, barrier, T, r, sigma)
            }
            
        except Exception as e:
            st.error(f"Error calculating barrier option: {str(e)}")
            return {'price': 0, 'greeks': {}, 'error': str(e)}
    
    def asian_option_price(self, S: float, K: float, T: float, r: float, 
                          sigma: float, num_observations: int = 252,
                          option_type: str = 'call', avg_type: str = 'arithmetic') -> Dict:
        """
        Price Asian options (average price options)
        
        Args:
            num_observations: Number of averaging observations
            avg_type: 'arithmetic' or 'geometric'
        """
        try:
            if avg_type == 'geometric':
                # Geometric Asian option has closed-form solution
                sigma_g = sigma / np.sqrt(3)
                mu_g = (r - 0.5 * sigma**2) / 2 + sigma_g**2 / 2
                
                d1 = (np.log(S/K) + (mu_g + 0.5*sigma_g**2)*T) / (sigma_g*np.sqrt(T))
                d2 = d1 - sigma_g*np.sqrt(T)
                
                if option_type == 'call':
                    price = np.exp(-r*T) * (S * np.exp(mu_g*T) * norm.cdf(d1) - K * norm.cdf(d2))
                else:
                    price = np.exp(-r*T) * (K * norm.cdf(-d2) - S * np.exp(mu_g*T) * norm.cdf(-d1))
                    
            else:
                # Arithmetic Asian option - use Monte Carlo
                price = self._monte_carlo_asian(S, K, T, r, sigma, num_observations, option_type)
            
            return {
                'price': max(0, price),
                'type': f"{avg_type}_asian_{option_type}",
                'averaging_observations': num_observations,
                'asian_volatility_adjustment': sigma / np.sqrt(3) if avg_type == 'geometric' else None
            }
            
        except Exception as e:
            st.error(f"Error calculating Asian option: {str(e)}")
            return {'price': 0, 'error': str(e)}
    
    def lookback_option_price(self, S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: str = 'call',
                             lookback_type: str = 'floating') -> Dict:
        """
        Price lookback options (path-dependent on max/min)
        """
        try:
            if lookback_type == 'floating':
                # Floating strike lookback
                if option_type == 'call':
                    # Payoff = S_T - min(S_t)
                    a1 = (np.log(S/S) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    a2 = a1 - sigma*np.sqrt(T)
                    
                    price = S * norm.cdf(a1) - S * np.exp(-r*T) * (sigma**2/(2*r)) * (
                        norm.cdf(a1) - np.exp(2*r*np.log(S/S)/sigma**2) * norm.cdf(a1 - 2*r*np.sqrt(T)/sigma)
                    )
                else:
                    # Payoff = max(S_t) - S_T
                    price = self._monte_carlo_lookback(S, K, T, r, sigma, option_type, lookback_type)
            else:
                # Fixed strike lookback
                price = self._monte_carlo_lookback(S, K, T, r, sigma, option_type, lookback_type)
            
            return {
                'price': max(0, price),
                'type': f"{lookback_type}_lookback_{option_type}",
                'path_dependency': 'extreme_value'
            }
            
        except Exception as e:
            st.error(f"Error calculating lookback option: {str(e)}")
            return {'price': 0, 'error': str(e)}
    
    def digital_option_price(self, S: float, K: float, T: float, r: float, 
                            sigma: float, payout: float = 1.0,
                            option_type: str = 'call') -> Dict:
        """
        Price digital (binary) options
        """
        try:
            d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            
            if option_type == 'call':
                price = payout * np.exp(-r*T) * norm.cdf(d2)
                delta = payout * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
            else:
                price = payout * np.exp(-r*T) * norm.cdf(-d2)
                delta = -payout * np.exp(-r*T) * norm.pdf(-d2) / (S * sigma * np.sqrt(T))
            
            return {
                'price': price,
                'delta': delta,
                'payout': payout,
                'type': f"digital_{option_type}",
                'probability_itm': norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)
            }
            
        except Exception as e:
            st.error(f"Error calculating digital option: {str(e)}")
            return {'price': 0, 'error': str(e)}
    
    def rainbow_option_price(self, S1: float, S2: float, K: float, T: float, 
                            r: float, sigma1: float, sigma2: float, 
                            correlation: float, option_type: str = 'max') -> Dict:
        """
        Price rainbow options (multi-asset options)
        """
        try:
            # Use Monte Carlo for rainbow options
            num_sims = 100000
            dt = T / 252
            
            # Generate correlated random walks
            z1 = np.random.normal(0, 1, (num_sims, int(T * 252)))
            z2 = correlation * z1 + np.sqrt(1 - correlation**2) * np.random.normal(0, 1, (num_sims, int(T * 252)))
            
            # Asset price paths
            S1_paths = S1 * np.exp(np.cumsum((r - 0.5*sigma1**2)*dt + sigma1*np.sqrt(dt)*z1, axis=1))
            S2_paths = S2 * np.exp(np.cumsum((r - 0.5*sigma2**2)*dt + sigma2*np.sqrt(dt)*z2, axis=1))
            
            # Final asset prices
            S1_final = S1_paths[:, -1]
            S2_final = S2_paths[:, -1]
            
            # Calculate payoffs
            if option_type == 'max':
                payoffs = np.maximum(np.maximum(S1_final, S2_final) - K, 0)
            elif option_type == 'min':
                payoffs = np.maximum(np.minimum(S1_final, S2_final) - K, 0)
            elif option_type == 'spread':
                payoffs = np.maximum(S1_final - S2_final - K, 0)
            
            price = np.exp(-r*T) * np.mean(payoffs)
            
            return {
                'price': price,
                'type': f"rainbow_{option_type}",
                'correlation': correlation,
                'asset1_weight': 0.5,
                'asset2_weight': 0.5,
                'monte_carlo_error': np.std(payoffs) / np.sqrt(num_sims)
            }
            
        except Exception as e:
            st.error(f"Error calculating rainbow option: {str(e)}")
            return {'price': 0, 'error': str(e)}
    
    def _calculate_barrier_greeks(self, S: float, K: float, T: float, r: float, 
                                 sigma: float, barrier: float, price: float) -> Dict:
        """Calculate Greeks for barrier options using finite differences"""
        try:
            h = 0.01
            
            # Delta
            price_up = self.barrier_option_price(S + h, K, T, r, sigma, barrier)['price']
            price_down = self.barrier_option_price(S - h, K, T, r, sigma, barrier)['price']
            delta = (price_up - price_down) / (2 * h)
            
            # Gamma
            gamma = (price_up - 2*price + price_down) / (h**2)
            
            # Theta
            price_t = self.barrier_option_price(S, K, T - 1/365, r, sigma, barrier)['price']
            theta = -(price_t - price) * 365
            
            # Vega
            price_vol = self.barrier_option_price(S, K, T, r, sigma + 0.01, barrier)['price']
            vega = (price_vol - price) / 0.01
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _barrier_touch_probability(self, S: float, barrier: float, T: float, 
                                  r: float, sigma: float) -> float:
        """Calculate probability of touching barrier"""
        try:
            mu = r - 0.5 * sigma**2
            if S > barrier:
                # Probability of down-crossing
                lambda_val = -2 * mu / sigma**2
                return (S/barrier)**lambda_val
            else:
                # Probability of up-crossing
                lambda_val = 2 * mu / sigma**2
                return (S/barrier)**lambda_val
        except:
            return 0.5
    
    def _monte_carlo_asian(self, S: float, K: float, T: float, r: float, 
                          sigma: float, num_obs: int, option_type: str) -> float:
        """Monte Carlo pricing for arithmetic Asian options"""
        num_sims = 50000
        dt = T / num_obs
        
        payoffs = []
        for _ in range(num_sims):
            path = [S]
            for _ in range(num_obs):
                dW = np.random.normal(0, np.sqrt(dt))
                path.append(path[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW))
            
            avg_price = np.mean(path[1:])  # Exclude initial price
            
            if option_type == 'call':
                payoff = max(avg_price - K, 0)
            else:
                payoff = max(K - avg_price, 0)
            
            payoffs.append(payoff)
        
        return np.exp(-r*T) * np.mean(payoffs)
    
    def _monte_carlo_lookback(self, S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: str, lookback_type: str) -> float:
        """Monte Carlo pricing for lookback options"""
        num_sims = 50000
        num_steps = 252
        dt = T / num_steps
        
        payoffs = []
        for _ in range(num_sims):
            path = [S]
            for _ in range(num_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                path.append(path[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW))
            
            if lookback_type == 'floating':
                if option_type == 'call':
                    payoff = path[-1] - min(path)
                else:
                    payoff = max(path) - path[-1]
            else:  # fixed
                if option_type == 'call':
                    payoff = max(max(path) - K, 0)
                else:
                    payoff = max(K - min(path), 0)
            
            payoffs.append(payoff)
        
        return np.exp(-r*T) * np.mean(payoffs)


class StructuredProducts:
    """Structured products and complex derivatives"""
    
    def __init__(self):
        self.exotic_engine = ExoticOptionsEngine()
    
    def autocallable_note(self, S: float, notional: float, coupon_rate: float,
                         barrier_level: float, observation_dates: List[float],
                         protection_level: float = 0.7) -> Dict:
        """
        Price autocallable structured note
        """
        try:
            # Simplified autocallable pricing
            # Early redemption probability at each observation
            redemption_probs = []
            pv_coupons = 0
            
            for i, obs_date in enumerate(observation_dates):
                # Probability of being above barrier at observation
                d = (np.log(S / barrier_level) + (0.05 - 0.5*0.25**2)*obs_date) / (0.25*np.sqrt(obs_date))
                prob_above = norm.cdf(d)
                
                # Probability of first redemption at this date
                prob_redemption = prob_above * np.prod([1 - p for p in redemption_probs])
                redemption_probs.append(prob_redemption)
                
                # Present value of coupon if redeemed
                pv_coupons += prob_redemption * coupon_rate * np.exp(-0.05 * obs_date)
            
            # Protection value
            prob_no_redemption = 1 - sum(redemption_probs)
            protection_value = prob_no_redemption * protection_level * np.exp(-0.05 * max(observation_dates))
            
            total_value = notional + pv_coupons + protection_value
            
            return {
                'value': total_value,
                'coupon_pv': pv_coupons,
                'protection_value': protection_value,
                'redemption_probability': sum(redemption_probs),
                'yield_to_call': (total_value - notional) / notional * (1 / min(observation_dates))
            }
            
        except Exception as e:
            return {'value': notional, 'error': str(e)}
    
    def reverse_convertible_note(self, S: float, K: float, T: float, 
                                coupon_rate: float, notional: float = 100) -> Dict:
        """
        Price reverse convertible note
        """
        try:
            # RCN = Zero-coupon bond + Short put option
            bond_value = notional * np.exp(-0.05 * T)  # Assume 5% risk-free rate
            
            # Put option value (short position)
            from models.black_scholes import BlackScholesModel
            bs = BlackScholesModel()
            put_value = bs.option_price(S, K, T, 0.05, 0.25, 'put')
            
            # Coupon payments
            coupon_pv = coupon_rate * notional * T * np.exp(-0.05 * T/2)
            
            rcn_value = bond_value + coupon_pv - put_value
            
            return {
                'value': rcn_value,
                'bond_component': bond_value,
                'coupon_component': coupon_pv,
                'option_component': -put_value,
                'yield': (rcn_value - notional) / notional / T,
                'protection_level': K / S
            }
            
        except Exception as e:
            return {'value': notional, 'error': str(e)}


# Initialize exotic options engine
@st.cache_resource
def get_exotic_engine():
    return ExoticOptionsEngine()

@st.cache_resource  
def get_structured_products():
    return StructuredProducts()