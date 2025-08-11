"""
Cryptocurrency Derivatives Pricing
Advanced models for crypto options and futures
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar, fsolve
import streamlit as st
from typing import Dict, List, Tuple, Optional
import yfinance as yf

class CryptoDerivativesEngine:
    """Cryptocurrency derivatives pricing with unique market characteristics"""
    
    def __init__(self):
        self.name = "Crypto Derivatives Engine"
        self.crypto_symbols = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD', 
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
            'MATIC': 'MATIC-USD',
            'DOT': 'DOT-USD',
            'AVAX': 'AVAX-USD',
            'LINK': 'LINK-USD'
        }
    
    def get_crypto_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Fetch cryptocurrency data with 24/7 market adjustments"""
        try:
            if symbol not in self.crypto_symbols:
                symbol = f"{symbol}-USD"
            else:
                symbol = self.crypto_symbols[symbol]
                
            data = yf.download(symbol, period=period, interval='1d')
            
            # Calculate crypto-specific metrics
            data['Returns'] = data['Close'].pct_change()
            data['Volatility_30d'] = data['Returns'].rolling(30).std() * np.sqrt(365)  # Annualized for 24/7
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['Fear_Greed_Proxy'] = self._fear_greed_indicator(data)
            
            return data.dropna()
            
        except Exception as e:
            st.error(f"Error fetching crypto data: {str(e)}")
            return pd.DataFrame()
    
    def crypto_option_price(self, S: float, K: float, T: float, r: float,
                           sigma: float, option_type: str = 'call',
                           crypto_adjustments: bool = True) -> Dict:
        """
        Price crypto options with market-specific adjustments
        
        Args:
            crypto_adjustments: Apply crypto-specific volatility and risk adjustments
        """
        try:
            if crypto_adjustments:
                # Crypto market adjustments
                sigma_adj = self._adjust_crypto_volatility(sigma, T)
                r_adj = self._adjust_risk_free_rate(r, T)  # Higher risk premium for crypto
                
                # Jump-diffusion component for crypto flash crashes
                jump_component = self._crypto_jump_adjustment(S, K, T, sigma_adj)
            else:
                sigma_adj = sigma
                r_adj = r
                jump_component = 0
            
            # Standard Black-Scholes with adjustments
            d1 = (np.log(S/K) + (r_adj + 0.5*sigma_adj**2)*T) / (sigma_adj*np.sqrt(T))
            d2 = d1 - sigma_adj*np.sqrt(T)
            
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r_adj*T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r_adj*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            # Add jump component
            price += jump_component
            
            # Calculate crypto-specific Greeks
            greeks = self._calculate_crypto_greeks(S, K, T, r_adj, sigma_adj, option_type)
            
            # Crypto market sentiment impact
            sentiment_impact = self._sentiment_adjustment(price, S, K)
            
            return {
                'price': max(0, price + sentiment_impact),
                'adjusted_volatility': sigma_adj,
                'adjusted_rate': r_adj,
                'jump_component': jump_component,
                'sentiment_impact': sentiment_impact,
                'greeks': greeks,
                'fear_greed_level': self._estimate_fear_greed()
            }
            
        except Exception as e:
            st.error(f"Error calculating crypto option: {str(e)}")
            return {'price': 0, 'error': str(e)}
    
    def perpetual_futures_price(self, S: float, funding_rate: float = 0.01,
                               premium_index: float = 0.0) -> Dict:
        """
        Price crypto perpetual futures contracts
        
        Args:
            funding_rate: 8-hour funding rate (typical 0.01% = 0.0001)
            premium_index: Premium/discount to spot price
        """
        try:
            # Perpetual futures fair value
            fair_value = S * (1 + premium_index)
            
            # Funding payment calculation (8-hour cycle)
            funding_payment = fair_value * funding_rate
            
            # Mark price convergence
            time_to_funding = 8  # hours
            convergence_factor = np.exp(-time_to_funding / 24)  # Daily decay
            
            mark_price = fair_value + (S - fair_value) * convergence_factor
            
            # Calculate basis
            basis = mark_price - S
            annualized_basis = basis / S * (365 * 3)  # 3 funding payments per day
            
            return {
                'mark_price': mark_price,
                'fair_value': fair_value,
                'basis': basis,
                'annualized_basis': annualized_basis,
                'funding_rate': funding_rate,
                'funding_payment': funding_payment,
                'time_to_funding': time_to_funding,
                'convergence_strength': 1 - convergence_factor
            }
            
        except Exception as e:
            st.error(f"Error calculating perpetual futures: {str(e)}")
            return {'mark_price': S, 'error': str(e)}
    
    def defi_option_price(self, S: float, K: float, T: float, protocol_risk: float = 0.05,
                         liquidity_premium: float = 0.02, gas_cost: float = 50) -> Dict:
        """
        Price DeFi options with protocol-specific risks
        
        Args:
            protocol_risk: Additional risk premium for smart contract risk
            liquidity_premium: Premium for lower liquidity
            gas_cost: Gas cost in USD for option exercise
        """
        try:
            # Base option price using crypto-adjusted Black-Scholes
            base_option = self.crypto_option_price(S, K, T, 0.05, 0.8, 'call', True)
            base_price = base_option['price']
            
            # DeFi-specific adjustments
            protocol_adjustment = base_price * protocol_risk * T  # Time-dependent protocol risk
            liquidity_adjustment = base_price * liquidity_premium
            
            # Gas cost impact on exercise decision
            gas_impact = gas_cost / S  # Relative gas cost
            exercise_threshold = K + gas_cost  # Effective strike including gas
            
            # Adjusted option value
            adjusted_price = base_price + protocol_adjustment + liquidity_adjustment
            
            # Calculate break-even levels
            breakeven_call = exercise_threshold + adjusted_price
            
            return {
                'price': adjusted_price,
                'base_price': base_price,
                'protocol_risk_premium': protocol_adjustment,
                'liquidity_premium': liquidity_adjustment,
                'gas_cost': gas_cost,
                'effective_strike': exercise_threshold,
                'breakeven_price': breakeven_call,
                'risk_metrics': {
                    'smart_contract_risk': protocol_risk,
                    'liquidity_risk': liquidity_premium,
                    'gas_risk': gas_impact
                }
            }
            
        except Exception as e:
            st.error(f"Error calculating DeFi option: {str(e)}")
            return {'price': 0, 'error': str(e)}
    
    def yield_farming_strategy(self, token_prices: Dict[str, float], 
                              pool_reserves: Dict[str, float],
                              apy: float, impermanent_loss_risk: float = 0.15) -> Dict:
        """
        Analyze yield farming strategy with options hedging
        """
        try:
            # Calculate impermanent loss
            if len(token_prices) == 2:
                tokens = list(token_prices.keys())
                p1, p2 = token_prices[tokens[0]], token_prices[tokens[1]]
                r1, r2 = pool_reserves[tokens[0]], pool_reserves[tokens[1]]
                
                # Price ratio change
                initial_ratio = r1 / r2
                current_ratio = p1 / p2
                ratio_change = current_ratio / initial_ratio
                
                # Impermanent loss calculation
                il = 2 * np.sqrt(ratio_change) / (1 + ratio_change) - 1
                il_usd = abs(il) * sum(token_prices.values()) * 0.5  # 50% allocation each
                
            else:
                il = 0
                il_usd = 0
            
            # Hedging strategy with put options
            hedge_cost = 0
            for token, price in token_prices.items():
                # Buy protective puts
                put_price = self.crypto_option_price(
                    price, price * 0.9, 0.25, 0.05, 0.8, 'put'
                )['price']
                hedge_cost += put_price * pool_reserves.get(token, 0)
            
            # Net yield calculation
            gross_yield = apy / 4  # Quarterly
            net_yield = gross_yield - hedge_cost - abs(il_usd)
            
            return {
                'gross_apy': apy,
                'impermanent_loss': il,
                'impermanent_loss_usd': il_usd,
                'hedge_cost': hedge_cost,
                'net_yield': net_yield,
                'risk_adjusted_apy': (net_yield * 4) if net_yield > 0 else 0,
                'tokens': list(token_prices.keys()),
                'hedging_recommendation': 'Recommended' if hedge_cost < gross_yield * 0.3 else 'Not Cost Effective'
            }
            
        except Exception as e:
            st.error(f"Error analyzing yield farming: {str(e)}")
            return {'gross_apy': apy, 'net_yield': 0, 'error': str(e)}
    
    def _adjust_crypto_volatility(self, sigma: float, T: float) -> float:
        """Adjust volatility for crypto market characteristics"""
        # Higher volatility for shorter terms (crypto flash crashes)
        term_adjustment = 1 + 0.5 * np.exp(-T * 4)  # Exponential decay
        
        # Weekend/24-7 trading adjustment
        weekend_factor = 1.2  # Crypto trades 24/7 including weekends
        
        return sigma * term_adjustment * weekend_factor
    
    def _adjust_risk_free_rate(self, r: float, T: float) -> float:
        """Adjust risk-free rate for crypto risk premium"""
        # Add crypto risk premium
        crypto_premium = 0.02 * (1 + np.exp(-T * 2))  # Higher for shorter terms
        return r + crypto_premium
    
    def _crypto_jump_adjustment(self, S: float, K: float, T: float, sigma: float) -> float:
        """Add jump component for crypto flash crashes"""
        # Simplified jump-diffusion adjustment
        jump_intensity = 2.0  # Expected 2 jumps per year
        jump_size = -0.15  # Average 15% down jump
        jump_vol = 0.3
        
        # Jump component value
        lambda_t = jump_intensity * T
        jump_component = -lambda_t * S * np.exp(jump_size) * np.exp(-0.05 * T)
        
        # Only apply for out-of-the-money puts (downside protection)
        if K < S:
            return jump_component * 0.1  # Small adjustment
        return 0
    
    def _calculate_crypto_greeks(self, S: float, K: float, T: float, r: float, 
                                sigma: float, option_type: str) -> Dict:
        """Calculate Greeks adjusted for crypto markets"""
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                delta = norm.cdf(d1)
                theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                        r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
            else:
                delta = norm.cdf(d1) - 1
                theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + 
                        r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
            
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            rho = K * T * np.exp(-r*T) * (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)) / 100
            
            # Crypto-specific adjustments
            delta *= 1.2  # Higher delta sensitivity due to volatility
            gamma *= 1.5  # Higher gamma due to price swings
            vega *= 1.3   # Higher vega sensitivity
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def _sentiment_adjustment(self, price: float, S: float, K: float) -> float:
        """Adjust option price based on market sentiment"""
        # Simplified sentiment model
        fear_greed = self._estimate_fear_greed()
        
        if fear_greed < 25:  # Extreme fear
            return price * 0.1  # Puts more expensive, calls cheaper
        elif fear_greed > 75:  # Extreme greed
            return price * -0.05  # Calls more expensive, puts cheaper
        
        return 0
    
    def _estimate_fear_greed(self) -> float:
        """Estimate Fear & Greed index (simplified)"""
        # In real implementation, would use actual Fear & Greed API
        return np.random.uniform(20, 80)  # Placeholder
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _fear_greed_indicator(self, data: pd.DataFrame) -> pd.Series:
        """Create fear/greed proxy from price data"""
        # Combine multiple indicators
        volatility_score = (1 - data['Volatility_30d'].rolling(30).rank(pct=True)) * 100
        momentum_score = data['Returns'].rolling(7).sum().rolling(30).rank(pct=True) * 100
        
        # Simple average (in real implementation, would use weighted factors)
        return (volatility_score + momentum_score) / 2


class NFTDerivatives:
    """NFT-based derivatives and floor price options"""
    
    def __init__(self):
        self.name = "NFT Derivatives"
    
    def nft_floor_option(self, current_floor: float, strike_floor: float, 
                        T: float, collection_volume: float,
                        rarity_factor: float = 1.0) -> Dict:
        """
        Price NFT floor price options
        
        Args:
            current_floor: Current floor price in ETH
            strike_floor: Strike floor price
            collection_volume: 30-day trading volume
            rarity_factor: Rarity multiplier (1.0 = floor, >1 = rare traits)
        """
        try:
            # NFT-specific volatility model
            volume_vol = np.log(collection_volume + 1) * 0.1  # Volume-based volatility
            base_vol = 1.5  # High base volatility for NFTs
            total_vol = base_vol + volume_vol
            
            # Liquidity adjustment
            liquidity_discount = min(0.3, 1 / (collection_volume + 1))
            
            # Simple option pricing with high volatility
            d1 = (np.log(current_floor/strike_floor) + (0.1 + 0.5*total_vol**2)*T) / (total_vol*np.sqrt(T))
            d2 = d1 - total_vol*np.sqrt(T)
            
            option_price = current_floor * norm.cdf(d1) - strike_floor * np.exp(-0.1*T) * norm.cdf(d2)
            
            # Apply rarity and liquidity adjustments
            adjusted_price = option_price * rarity_factor * (1 - liquidity_discount)
            
            return {
                'price': max(0, adjusted_price),
                'floor_price': current_floor,
                'strike_floor': strike_floor,
                'volatility': total_vol,
                'rarity_factor': rarity_factor,
                'liquidity_discount': liquidity_discount,
                'collection_volume': collection_volume
            }
            
        except Exception as e:
            return {'price': 0, 'error': str(e)}


# Initialize crypto derivatives engines
@st.cache_resource
def get_crypto_engine():
    return CryptoDerivativesEngine()

@st.cache_resource
def get_nft_derivatives():
    return NFTDerivatives()