import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
import streamlit as st
from typing import Dict, Tuple, List, Optional

class VolatilityModels:
    """Collection of volatility modeling approaches"""
    
    @staticmethod
    def realized_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling realized volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def ewma_volatility(returns: pd.Series, lambda_param: float = 0.94) -> pd.Series:
        """Exponentially Weighted Moving Average volatility"""
        weights = [(1 - lambda_param) * (lambda_param ** i) for i in range(len(returns))]
        weights = np.array(weights[::-1])
        weights = weights / weights.sum()
        
        ewma_var = []
        for i in range(len(returns)):
            if i == 0:
                ewma_var.append(returns.iloc[i] ** 2)
            else:
                current_weights = weights[:i+1]
                current_weights = current_weights / current_weights.sum()
                var = np.sum(current_weights * (returns.iloc[:i+1] ** 2))
                ewma_var.append(var)
        
        return pd.Series(np.sqrt(ewma_var) * np.sqrt(252), index=returns.index)

class GARCHModel:
    """GARCH volatility modeling"""
    
    def __init__(self, returns: pd.Series):
        self.returns = returns
        self.model = None
        self.fitted_model = None
    
    def fit_garch(self, p: int = 1, q: int = 1, dist: str = 'normal') -> Dict:
        """Fit GARCH(p,q) model"""
        try:
            # Prepare returns (remove any NaN values)
            clean_returns = self.returns.dropna()
            
            # Fit GARCH model
            self.model = arch_model(clean_returns, vol='Garch', p=p, q=q, dist=dist)
            self.fitted_model = self.model.fit(disp='off')
            
            return {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.loglikelihood,
                'params': self.fitted_model.params.to_dict(),
                'summary': str(self.fitted_model.summary())
            }
        except Exception as e:
            st.error(f"Error fitting GARCH model: {str(e)}")
            return {}
    
    def forecast_volatility(self, horizon: int = 30) -> Dict:
        """Forecast volatility using fitted GARCH model"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.fitted_model.forecast(horizon=horizon, reindex=False)
        
        return {
            'mean_forecast': forecast.mean.iloc[-horizon:].values,
            'variance_forecast': forecast.variance.iloc[-horizon:].values,
            'volatility_forecast': np.sqrt(forecast.variance.iloc[-horizon:].values * 252),
            'residual_variance': forecast.residual_variance.iloc[-horizon:].values if hasattr(forecast, 'residual_variance') else None
        }
    
    def conditional_volatility(self) -> pd.Series:
        """Extract conditional volatility from fitted model"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.conditional_volatility

class ImpliedVolatilitySurface:
    """Implied volatility surface construction and analysis"""
    
    def __init__(self):
        self.surface_data = None
        self.calibrated_params = None
    
    def create_synthetic_surface(self, S: float, strikes: List[float], 
                                expiries: List[float], base_vol: float = 0.2) -> pd.DataFrame:
        """Create synthetic implied volatility surface"""
        surface_data = []
        
        for T in expiries:
            for K in strikes:
                # Volatility smile/skew pattern
                moneyness = K / S
                
                # Volatility smile: higher vol for OTM options
                if moneyness < 0.95:  # ITM puts / OTM calls
                    vol_smile = base_vol + 0.05 * (0.95 - moneyness)
                elif moneyness > 1.05:  # OTM puts / ITM calls
                    vol_smile = base_vol + 0.03 * (moneyness - 1.05)
                else:  # ATM
                    vol_smile = base_vol
                
                # Term structure: vol tends to be higher for shorter terms
                term_adjustment = base_vol * (1 + 0.1 * np.exp(-T * 2))
                
                # Combine effects
                implied_vol = vol_smile + term_adjustment * 0.1
                
                surface_data.append({
                    'Strike': K,
                    'Expiry': T,
                    'Moneyness': moneyness,
                    'ImpliedVol': implied_vol,
                    'Spot': S
                })
        
        self.surface_data = pd.DataFrame(surface_data)
        return self.surface_data
    
    def calibrate_svi_model(self, market_data: pd.DataFrame) -> Dict:
        """Calibrate SVI (Stochastic Volatility Inspired) model to market data"""
        
        def svi_function(k, a, b, rho, m, sigma):
            """SVI parameterization for implied variance"""
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        def objective_function(params, k_values, market_vols):
            """Objective function for SVI calibration"""
            a, b, rho, m, sigma = params
            model_vars = [svi_function(k, a, b, rho, m, sigma) for k in k_values]
            market_vars = [vol**2 for vol in market_vols]
            return np.sum([(mv - model_var)**2 for mv, model_var in zip(market_vars, model_vars)])
        
        calibration_results = {}
        
        # Group by expiry and calibrate SVI for each tenor
        for expiry in market_data['Expiry'].unique():
            tenor_data = market_data[market_data['Expiry'] == expiry].copy()
            tenor_data['LogMoneyness'] = np.log(tenor_data['Moneyness'])
            
            k_values = tenor_data['LogMoneyness'].values
            vol_values = tenor_data['ImpliedVol'].values
            
            # Initial parameter guess
            initial_params = [0.04, 0.1, -0.5, 0.0, 0.1]
            
            # Parameter bounds
            bounds = [(-1, 1), (0, 1), (-1, 1), (-1, 1), (0.01, 1)]
            
            try:
                result = minimize(objective_function, initial_params, 
                                args=(k_values, vol_values), bounds=bounds, 
                                method='L-BFGS-B')
                
                calibration_results[expiry] = {
                    'params': result.x,
                    'success': result.success,
                    'residual': result.fun,
                    'param_names': ['a', 'b', 'rho', 'm', 'sigma']
                }
            except Exception as e:
                st.warning(f"SVI calibration failed for expiry {expiry}: {str(e)}")
                calibration_results[expiry] = None
        
        self.calibrated_params = calibration_results
        return calibration_results
    
    def interpolate_surface(self, target_strikes: List[float], 
                           target_expiries: List[float]) -> pd.DataFrame:
        """Interpolate volatility surface to new strikes and expiries"""
        if self.surface_data is None:
            raise ValueError("Surface data must be created first")
        
        from scipy.interpolate import griddata
        
        # Extract existing surface points
        points = self.surface_data[['Strike', 'Expiry']].values
        values = self.surface_data['ImpliedVol'].values
        
        # Create new grid
        new_points = [(s, t) for t in target_expiries for s in target_strikes]
        
        # Interpolate
        interpolated_vols = griddata(points, values, new_points, method='cubic', 
                                   fill_value=np.nan)
        
        # Create interpolated surface DataFrame
        interpolated_data = []
        idx = 0
        for T in target_expiries:
            for K in target_strikes:
                interpolated_data.append({
                    'Strike': K,
                    'Expiry': T,
                    'ImpliedVol': interpolated_vols[idx],
                    'Moneyness': K / self.surface_data['Spot'].iloc[0]
                })
                idx += 1
        
        return pd.DataFrame(interpolated_data)

class StochasticVolatilityModels:
    """Advanced stochastic volatility models"""
    
    @staticmethod
    def heston_characteristic_function(phi: complex, S0: float, v0: float, 
                                     kappa: float, theta: float, sigma: float, 
                                     rho: float, r: float, T: float) -> complex:
        """Heston model characteristic function"""
        # Implementation from models/black_scholes.py HestonModel
        xi = kappa - rho * sigma * phi * 1j
        d = np.sqrt(xi**2 + sigma**2 * (phi * 1j + phi**2))
        
        A1 = phi * 1j * (np.log(S0) + r * T)
        A2 = (kappa * theta) / (sigma**2) * (xi - d) * T
        A3 = -(kappa * theta) / (sigma**2) * np.log((1 - (xi - d) * np.exp(-d * T) / (2 * d)) / 
                                                     (1 - (xi - d) / (2 * d)))
        A4 = -(v0 / sigma**2) * (xi - d) * (1 - np.exp(-d * T)) / (1 - (xi - d) * np.exp(-d * T) / (2 * d))
        
        return np.exp(A1 + A2 + A3 + A4)
    
    @staticmethod
    def calibrate_heston_to_surface(surface_data: pd.DataFrame, S0: float, r: float) -> Dict:
        """Calibrate Heston model parameters to implied volatility surface"""
        
        def heston_vol_approxmation(K: float, T: float, params: List[float]) -> float:
            """Approximate Heston implied volatility using closed-form approximation"""
            v0, kappa, theta, sigma, rho = params
            
            # Use Heston approximation formulas
            sqrt_v0 = np.sqrt(v0)
            sqrt_theta = np.sqrt(theta)
            
            # ATM volatility approximation
            atm_vol = sqrt_v0 + (sqrt_theta - sqrt_v0) * (1 - np.exp(-kappa * T)) / (kappa * T)
            
            # Skew adjustment
            moneyness = np.log(K / S0)
            skew = rho * sigma * sqrt_v0 * moneyness / atm_vol
            
            return atm_vol + skew
        
        def objective_function(params: List[float]) -> float:
            """Objective function for Heston calibration"""
            if any(p <= 0 for p in params[:4]) or abs(params[4]) >= 1:
                return 1e6  # Penalty for invalid parameters
            
            total_error = 0
            for _, row in surface_data.iterrows():
                model_vol = heston_vol_approxmation(row['Strike'], row['Expiry'], params)
                market_vol = row['ImpliedVol']
                total_error += (model_vol - market_vol) ** 2
            
            return total_error
        
        # Initial parameter guess [v0, kappa, theta, sigma, rho]
        initial_params = [0.04, 2.0, 0.04, 0.3, -0.7]
        
        # Parameter bounds
        bounds = [(0.001, 1.0), (0.1, 10.0), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.99)]
        
        try:
            result = minimize(objective_function, initial_params, bounds=bounds, 
                            method='L-BFGS-B')
            
            return {
                'params': {
                    'v0': result.x[0],
                    'kappa': result.x[1], 
                    'theta': result.x[2],
                    'sigma': result.x[3],
                    'rho': result.x[4]
                },
                'success': result.success,
                'residual': result.fun,
                'message': result.message
            }
        except Exception as e:
            st.error(f"Heston calibration failed: {str(e)}")
            return {}
