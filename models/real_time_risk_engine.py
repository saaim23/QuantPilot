"""
Real-Time Risk Management Engine
Advanced risk monitoring and alert system
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealTimeRiskEngine:
    """Real-time portfolio risk monitoring and management"""
    
    def __init__(self):
        self.name = "Real-Time Risk Engine"
        self.risk_limits = {
            'max_portfolio_var': 0.05,  # 5% daily VaR
            'max_position_size': 0.10,   # 10% max position
            'max_sector_exposure': 0.25, # 25% max sector
            'min_liquidity_ratio': 0.20  # 20% cash/liquid assets
        }
        self.alert_history = []
    
    def real_time_risk_monitor(self, portfolio: Dict, market_data: pd.DataFrame,
                              confidence_level: float = 0.95) -> Dict:
        """
        Real-time portfolio risk monitoring with alerts
        
        Args:
            portfolio: Dictionary with positions and weights
            market_data: Real-time market data
            confidence_level: VaR confidence level
        """
        try:
            current_time = datetime.now()
            
            # Calculate portfolio metrics
            portfolio_value = sum(portfolio.get('values', [100000]))
            
            # Real-time VaR calculation
            var_1d = self._calculate_real_time_var(portfolio, market_data, confidence_level)
            var_10d = var_1d * np.sqrt(10)  # 10-day VaR
            
            # Stress testing
            stress_results = self._real_time_stress_test(portfolio, market_data)
            
            # Liquidity risk
            liquidity_risk = self._assess_liquidity_risk(portfolio, market_data)
            
            # Concentration risk
            concentration_risk = self._calculate_concentration_risk(portfolio)
            
            # Market regime detection
            regime = self._detect_market_regime(market_data)
            
            # Generate risk alerts
            alerts = self._generate_risk_alerts(
                var_1d, stress_results, liquidity_risk, concentration_risk, regime
            )
            
            # Risk attribution
            risk_attribution = self._calculate_risk_attribution(portfolio, market_data)
            
            # Dynamic hedging recommendations
            hedge_recommendations = self._generate_hedge_recommendations(
                portfolio, var_1d, regime
            )
            
            return {
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'var_1d': var_1d,
                'var_10d': var_10d,
                'var_percentage': (var_1d / portfolio_value) * 100,
                'stress_test_results': stress_results,
                'liquidity_score': liquidity_risk['score'],
                'concentration_score': concentration_risk['score'],
                'market_regime': regime,
                'risk_alerts': alerts,
                'risk_attribution': risk_attribution,
                'hedge_recommendations': hedge_recommendations,
                'overall_risk_score': self._calculate_overall_risk_score(
                    var_1d, stress_results, liquidity_risk, concentration_risk
                )
            }
            
        except Exception as e:
            st.error(f"Risk monitoring error: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def dynamic_position_sizing(self, symbol: str, current_volatility: float,
                               target_risk: float, portfolio_value: float) -> Dict:
        """
        Calculate optimal position size based on risk targets
        
        Args:
            symbol: Asset symbol
            current_volatility: Asset volatility
            target_risk: Target risk percentage
            portfolio_value: Total portfolio value
        """
        try:
            # Kelly Criterion with modifications
            win_rate = 0.55  # Estimated win rate
            avg_win = 0.02   # Average win percentage
            avg_loss = 0.015  # Average loss percentage
            
            # Kelly fraction
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Risk-adjusted position size
            volatility_adjustment = 0.20 / max(current_volatility, 0.05)  # Scale by volatility
            
            # Position size calculation
            max_risk_per_trade = target_risk * portfolio_value
            position_size = min(
                kelly_fraction * volatility_adjustment * portfolio_value,
                max_risk_per_trade / (2 * current_volatility)  # 2 sigma stop loss
            )
            
            # Position size limits
            max_position = portfolio_value * 0.10  # 10% max
            min_position = portfolio_value * 0.001  # 0.1% min
            
            optimal_size = np.clip(position_size, min_position, max_position)
            
            return {
                'optimal_position_size': optimal_size,
                'position_percentage': (optimal_size / portfolio_value) * 100,
                'kelly_fraction': kelly_fraction,
                'volatility_adjustment': volatility_adjustment,
                'risk_per_trade': max_risk_per_trade,
                'stop_loss_level': 2 * current_volatility,
                'recommendation': self._position_size_recommendation(optimal_size, portfolio_value)
            }
            
        except Exception as e:
            return {'optimal_position_size': 0, 'error': str(e)}
    
    def _calculate_real_time_var(self, portfolio: Dict, data: pd.DataFrame, 
                                confidence: float) -> float:
        """Calculate real-time Value at Risk"""
        try:
            # Get portfolio weights and returns
            weights = np.array(portfolio.get('weights', [1.0]))
            
            if len(data) < 30:
                return 0  # Insufficient data
            
            # Calculate portfolio returns
            returns = data['Close'].pct_change().dropna()
            portfolio_returns = returns * weights[0] if len(weights) == 1 else returns
            
            # Exponentially weighted volatility
            lambda_decay = 0.94
            ewm_vol = portfolio_returns.ewm(alpha=1-lambda_decay).std()
            current_vol = ewm_vol.iloc[-1] if len(ewm_vol) > 0 else 0.02
            
            # VaR calculation
            z_score = stats.norm.ppf(1 - confidence)
            var_estimate = abs(z_score) * current_vol * np.sqrt(1)  # 1-day
            
            # Portfolio value
            portfolio_value = sum(portfolio.get('values', [100000]))
            
            return var_estimate * portfolio_value
            
        except Exception:
            return 0
    
    def _real_time_stress_test(self, portfolio: Dict, data: pd.DataFrame) -> Dict:
        """Real-time stress testing scenarios"""
        try:
            current_price = data['Close'].iloc[-1] if len(data) > 0 else 100
            portfolio_value = sum(portfolio.get('values', [100000]))
            
            # Stress scenarios
            scenarios = {
                'market_crash': -0.20,      # 20% market drop
                'flash_crash': -0.10,       # 10% flash crash
                'volatility_spike': -0.05,  # 5% with 2x volatility
                'liquidity_crisis': -0.15,  # 15% with liquidity issues
                'black_swan': -0.30         # 30% extreme event
            }
            
            stress_results = {}
            for scenario, shock in scenarios.items():
                # Apply shock to portfolio
                shocked_value = portfolio_value * (1 + shock)
                loss = portfolio_value - shocked_value
                
                # Additional scenario-specific factors
                if scenario == 'volatility_spike':
                    vol_impact = portfolio_value * 0.02  # Additional volatility cost
                    loss += vol_impact
                elif scenario == 'liquidity_crisis':
                    liquidity_cost = portfolio_value * 0.05  # Liquidity premium
                    loss += liquidity_cost
                
                stress_results[scenario] = {
                    'loss': loss,
                    'loss_percentage': (loss / portfolio_value) * 100,
                    'new_portfolio_value': portfolio_value - loss
                }
            
            return stress_results
            
        except Exception:
            return {}
    
    def _assess_liquidity_risk(self, portfolio: Dict, data: pd.DataFrame) -> Dict:
        """Assess portfolio liquidity risk"""
        try:
            # Simplified liquidity assessment
            volume = data['Volume'].iloc[-20:].mean() if len(data) >= 20 else 1000000
            price = data['Close'].iloc[-1] if len(data) > 0 else 100
            
            # Market cap proxy
            market_cap = price * volume * 252  # Rough estimate
            
            # Liquidity score (0-1, higher is more liquid)
            if market_cap > 10e9:  # Large cap
                liquidity_score = 0.9
            elif market_cap > 1e9:  # Mid cap
                liquidity_score = 0.7
            elif market_cap > 100e6:  # Small cap
                liquidity_score = 0.5
            else:  # Micro cap
                liquidity_score = 0.3
            
            # Bid-ask spread proxy (using price volatility)
            volatility = data['Close'].pct_change().std() if len(data) > 1 else 0.02
            spread_estimate = volatility * 0.1  # Rough estimate
            
            return {
                'score': liquidity_score,
                'market_cap_estimate': market_cap,
                'spread_estimate': spread_estimate,
                'volume_20d_avg': volume,
                'liquidity_tier': self._get_liquidity_tier(liquidity_score)
            }
            
        except Exception:
            return {'score': 0.5, 'liquidity_tier': 'Unknown'}
    
    def _calculate_concentration_risk(self, portfolio: Dict) -> Dict:
        """Calculate portfolio concentration risk"""
        try:
            weights = np.array(portfolio.get('weights', [1.0]))
            
            # Herfindahl-Hirschman Index
            hhi = np.sum(weights ** 2)
            
            # Number of effective positions
            effective_positions = 1 / hhi if hhi > 0 else 1
            
            # Concentration score (0-1, lower is better)
            if effective_positions >= 20:
                concentration_score = 0.1  # Well diversified
            elif effective_positions >= 10:
                concentration_score = 0.3  # Moderately diversified
            elif effective_positions >= 5:
                concentration_score = 0.6  # Concentrated
            else:
                concentration_score = 0.9  # Highly concentrated
            
            return {
                'score': concentration_score,
                'hhi': hhi,
                'effective_positions': effective_positions,
                'largest_position': np.max(weights) if len(weights) > 0 else 0,
                'top_5_concentration': np.sum(np.sort(weights)[-5:]) if len(weights) >= 5 else np.sum(weights)
            }
            
        except Exception:
            return {'score': 0.5, 'effective_positions': 1}
    
    def _detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """Detect current market regime"""
        try:
            if len(data) < 50:
                return {'regime': 'Unknown', 'confidence': 0}
            
            returns = data['Close'].pct_change().dropna()
            
            # Calculate regime indicators
            volatility = returns.rolling(20).std().iloc[-1]
            trend = returns.rolling(20).mean().iloc[-1]
            momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) if len(data) >= 20 else 0
            
            # Regime classification
            if volatility > 0.03:  # High volatility
                if trend < -0.001:
                    regime = 'Crisis'
                    confidence = 0.8
                else:
                    regime = 'High Volatility'
                    confidence = 0.7
            elif volatility < 0.01:  # Low volatility
                if momentum > 0.05:
                    regime = 'Bull Market'
                    confidence = 0.7
                else:
                    regime = 'Low Volatility'
                    confidence = 0.6
            else:  # Normal volatility
                if trend > 0.001:
                    regime = 'Normal Bull'
                    confidence = 0.6
                elif trend < -0.001:
                    regime = 'Normal Bear'
                    confidence = 0.6
                else:
                    regime = 'Sideways'
                    confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility': volatility,
                'trend': trend,
                'momentum': momentum
            }
            
        except Exception:
            return {'regime': 'Unknown', 'confidence': 0}
    
    def _generate_risk_alerts(self, var_1d: float, stress_results: Dict,
                             liquidity_risk: Dict, concentration_risk: Dict,
                             regime: Dict) -> List[Dict]:
        """Generate risk alerts based on current conditions"""
        alerts = []
        current_time = datetime.now()
        
        # VaR alerts
        portfolio_value = 100000  # Default value
        var_percentage = (var_1d / portfolio_value) * 100
        
        if var_percentage > 5:
            alerts.append({
                'type': 'HIGH_RISK',
                'message': f'Portfolio VaR exceeds 5% threshold: {var_percentage:.2f}%',
                'severity': 'CRITICAL',
                'timestamp': current_time
            })
        elif var_percentage > 3:
            alerts.append({
                'type': 'MEDIUM_RISK',
                'message': f'Portfolio VaR elevated: {var_percentage:.2f}%',
                'severity': 'WARNING',
                'timestamp': current_time
            })
        
        # Concentration alerts
        if concentration_risk.get('score', 0) > 0.7:
            alerts.append({
                'type': 'CONCENTRATION',
                'message': f'High concentration risk detected',
                'severity': 'WARNING',
                'timestamp': current_time
            })
        
        # Liquidity alerts
        if liquidity_risk.get('score', 1) < 0.4:
            alerts.append({
                'type': 'LIQUIDITY',
                'message': 'Low liquidity positions detected',
                'severity': 'WARNING',
                'timestamp': current_time
            })
        
        # Market regime alerts
        if regime.get('regime') == 'Crisis':
            alerts.append({
                'type': 'MARKET_REGIME',
                'message': 'Crisis market regime detected',
                'severity': 'CRITICAL',
                'timestamp': current_time
            })
        
        # Stress test alerts
        worst_scenario = max(stress_results.values(), 
                           key=lambda x: x.get('loss_percentage', 0), 
                           default={'loss_percentage': 0})
        
        if worst_scenario.get('loss_percentage', 0) > 25:
            alerts.append({
                'type': 'STRESS_TEST',
                'message': f'Severe stress test loss: {worst_scenario["loss_percentage"]:.1f}%',
                'severity': 'CRITICAL',
                'timestamp': current_time
            })
        
        return alerts
    
    def _calculate_risk_attribution(self, portfolio: Dict, data: pd.DataFrame) -> Dict:
        """Calculate risk attribution by position/factor"""
        try:
            weights = np.array(portfolio.get('weights', [1.0]))
            
            # Simplified risk attribution
            total_risk = np.sum(weights ** 2) * 0.04  # Simplified calculation
            
            position_risks = []
            for i, weight in enumerate(weights):
                position_risk = (weight ** 2) * 0.04  # Individual position risk
                contribution = (position_risk / total_risk) * 100 if total_risk > 0 else 0
                
                position_risks.append({
                    'position': f'Position_{i+1}',
                    'weight': weight,
                    'risk_contribution': contribution,
                    'marginal_risk': position_risk
                })
            
            return {
                'total_portfolio_risk': total_risk,
                'position_contributions': position_risks,
                'largest_contributor': max(position_risks, key=lambda x: x['risk_contribution']) if position_risks else None
            }
            
        except Exception:
            return {'total_portfolio_risk': 0, 'position_contributions': []}
    
    def _generate_hedge_recommendations(self, portfolio: Dict, var_1d: float, 
                                      regime: Dict) -> List[Dict]:
        """Generate dynamic hedging recommendations"""
        recommendations = []
        
        try:
            portfolio_value = sum(portfolio.get('values', [100000]))
            var_percentage = (var_1d / portfolio_value) * 100
            
            # High VaR recommendations
            if var_percentage > 4:
                recommendations.append({
                    'type': 'VIX_HEDGE',
                    'action': 'Buy VIX calls',
                    'rationale': 'High portfolio VaR requires volatility hedge',
                    'allocation': min(0.05, var_percentage / 100),
                    'priority': 'HIGH'
                })
            
            # Market regime-based recommendations
            if regime.get('regime') == 'Crisis':
                recommendations.append({
                    'type': 'DEFENSIVE_HEDGE',
                    'action': 'Buy protective puts',
                    'rationale': 'Crisis regime detected',
                    'allocation': 0.10,
                    'priority': 'CRITICAL'
                })
            elif regime.get('regime') == 'High Volatility':
                recommendations.append({
                    'type': 'VOLATILITY_HEDGE',
                    'action': 'Sell volatility (careful)',
                    'rationale': 'High volatility may mean-revert',
                    'allocation': 0.02,
                    'priority': 'MEDIUM'
                })
            
            # Concentration-based recommendations
            weights = np.array(portfolio.get('weights', [1.0]))
            if len(weights) > 0 and np.max(weights) > 0.15:
                recommendations.append({
                    'type': 'DIVERSIFICATION',
                    'action': 'Reduce largest position',
                    'rationale': 'High concentration risk',
                    'allocation': -0.05,
                    'priority': 'MEDIUM'
                })
            
            return recommendations
            
        except Exception:
            return []
    
    def _calculate_overall_risk_score(self, var_1d: float, stress_results: Dict,
                                    liquidity_risk: Dict, concentration_risk: Dict) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        try:
            # Component scores (0-1)
            var_score = min(1.0, (var_1d / 5000))  # Normalize by $5k VaR
            
            stress_score = 0
            if stress_results:
                worst_loss = max([s.get('loss_percentage', 0) for s in stress_results.values()])
                stress_score = min(1.0, worst_loss / 30)  # Normalize by 30% loss
            
            liquidity_score = 1 - liquidity_risk.get('score', 0.5)
            concentration_score = concentration_risk.get('score', 0.5)
            
            # Weighted average
            weights = [0.4, 0.3, 0.15, 0.15]  # VaR, stress, liquidity, concentration
            scores = [var_score, stress_score, liquidity_score, concentration_score]
            
            overall_score = np.average(scores, weights=weights) * 100
            
            return min(100, max(0, overall_score))
            
        except Exception:
            return 50  # Default medium risk
    
    def _position_size_recommendation(self, position_size: float, portfolio_value: float) -> str:
        """Generate position sizing recommendation"""
        percentage = (position_size / portfolio_value) * 100
        
        if percentage < 1:
            return "CONSERVATIVE - Very small position"
        elif percentage < 3:
            return "MODERATE - Standard position size"
        elif percentage < 7:
            return "AGGRESSIVE - Large position"
        else:
            return "EXTREME - Consider reducing size"
    
    def _get_liquidity_tier(self, score: float) -> str:
        """Get liquidity tier description"""
        if score >= 0.8:
            return "Highly Liquid"
        elif score >= 0.6:
            return "Liquid"
        elif score >= 0.4:
            return "Moderately Liquid"
        else:
            return "Illiquid"


@st.cache_resource
def get_risk_engine():
    return RealTimeRiskEngine()