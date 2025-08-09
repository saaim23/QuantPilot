import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
from models.black_scholes import BlackScholesModel
from models.monte_carlo import MonteCarloEngine
from utils.calculations import FinancialCalculations, RiskMetrics
from data.market_data import MarketDataProvider
import warnings
warnings.filterwarnings('ignore')

class OptionTradingStrategy:
    """Option trading strategy implementation and backtesting"""
    
    def __init__(self):
        self.positions = []
        self.trades = []
        self.cash = 0
        self.portfolio_value = 0
        
    def long_call_strategy(self, S: float, K: float, T: float, r: float, sigma: float,
                          position_size: float) -> Dict[str, Any]:
        """Long call strategy"""
        
        option_price = BlackScholesModel.option_price(S, K, T, r, sigma, 'call')
        cost = option_price * position_size
        
        return {
            'strategy': 'long_call',
            'option_type': 'call',
            'position_type': 'long',
            'strike': K,
            'expiry': T,
            'premium': option_price,
            'quantity': position_size,
            'cost': cost,
            'max_profit': np.inf,
            'max_loss': cost,
            'breakeven': K + option_price
        }
    
    def long_put_strategy(self, S: float, K: float, T: float, r: float, sigma: float,
                         position_size: float) -> Dict[str, Any]:
        """Long put strategy"""
        
        option_price = BlackScholesModel.option_price(S, K, T, r, sigma, 'put')
        cost = option_price * position_size
        
        return {
            'strategy': 'long_put',
            'option_type': 'put',
            'position_type': 'long',
            'strike': K,
            'expiry': T,
            'premium': option_price,
            'quantity': position_size,
            'cost': cost,
            'max_profit': (K - option_price) * position_size,
            'max_loss': cost,
            'breakeven': K - option_price
        }
    
    def covered_call_strategy(self, S: float, K: float, T: float, r: float, sigma: float,
                             position_size: float) -> Dict[str, Any]:
        """Covered call strategy (long stock + short call)"""
        
        call_price = BlackScholesModel.option_price(S, K, T, r, sigma, 'call')
        stock_cost = S * position_size
        premium_received = call_price * position_size
        net_cost = stock_cost - premium_received
        
        return {
            'strategy': 'covered_call',
            'components': [
                {'type': 'stock', 'position': 'long', 'quantity': position_size, 'price': S},
                {'type': 'call', 'position': 'short', 'quantity': position_size, 'price': call_price}
            ],
            'strike': K,
            'expiry': T,
            'net_cost': net_cost,
            'max_profit': (K - S + call_price) * position_size,
            'max_loss': net_cost,
            'breakeven': S - call_price
        }
    
    def protective_put_strategy(self, S: float, K: float, T: float, r: float, sigma: float,
                               position_size: float) -> Dict[str, Any]:
        """Protective put strategy (long stock + long put)"""
        
        put_price = BlackScholesModel.option_price(S, K, T, r, sigma, 'put')
        stock_cost = S * position_size
        put_cost = put_price * position_size
        total_cost = stock_cost + put_cost
        
        return {
            'strategy': 'protective_put',
            'components': [
                {'type': 'stock', 'position': 'long', 'quantity': position_size, 'price': S},
                {'type': 'put', 'position': 'long', 'quantity': position_size, 'price': put_price}
            ],
            'strike': K,
            'expiry': T,
            'total_cost': total_cost,
            'max_profit': np.inf,
            'max_loss': total_cost - K * position_size,
            'breakeven': S + put_price
        }
    
    def iron_condor_strategy(self, S: float, put_strike_low: float, put_strike_high: float,
                            call_strike_low: float, call_strike_high: float,
                            T: float, r: float, sigma: float, position_size: float) -> Dict[str, Any]:
        """Iron condor strategy"""
        
        # Calculate option prices
        put_low_price = BlackScholesModel.option_price(S, put_strike_low, T, r, sigma, 'put')
        put_high_price = BlackScholesModel.option_price(S, put_strike_high, T, r, sigma, 'put')
        call_low_price = BlackScholesModel.option_price(S, call_strike_low, T, r, sigma, 'call')
        call_high_price = BlackScholesModel.option_price(S, call_strike_high, T, r, sigma, 'call')
        
        # Net credit received
        net_credit = (put_high_price - put_low_price + call_low_price - call_high_price) * position_size
        
        return {
            'strategy': 'iron_condor',
            'components': [
                {'type': 'put', 'position': 'long', 'strike': put_strike_low, 'price': put_low_price},
                {'type': 'put', 'position': 'short', 'strike': put_strike_high, 'price': put_high_price},
                {'type': 'call', 'position': 'short', 'strike': call_strike_low, 'price': call_low_price},
                {'type': 'call', 'position': 'long', 'strike': call_strike_high, 'price': call_high_price}
            ],
            'expiry': T,
            'quantity': position_size,
            'net_credit': net_credit,
            'max_profit': net_credit,
            'max_loss': (call_strike_high - call_strike_low) * position_size - net_credit,
            'breakeven_lower': put_strike_high - net_credit / position_size,
            'breakeven_upper': call_strike_low + net_credit / position_size
        }
    
    def straddle_strategy(self, S: float, K: float, T: float, r: float, sigma: float,
                         position_size: float, long: bool = True) -> Dict[str, Any]:
        """Long/Short straddle strategy"""
        
        call_price = BlackScholesModel.option_price(S, K, T, r, sigma, 'call')
        put_price = BlackScholesModel.option_price(S, K, T, r, sigma, 'put')
        total_premium = (call_price + put_price) * position_size
        
        if long:
            return {
                'strategy': 'long_straddle',
                'components': [
                    {'type': 'call', 'position': 'long', 'strike': K, 'price': call_price},
                    {'type': 'put', 'position': 'long', 'strike': K, 'price': put_price}
                ],
                'strike': K,
                'expiry': T,
                'quantity': position_size,
                'cost': total_premium,
                'max_profit': np.inf,
                'max_loss': total_premium,
                'breakeven_upper': K + (call_price + put_price),
                'breakeven_lower': K - (call_price + put_price)
            }
        else:
            return {
                'strategy': 'short_straddle',
                'components': [
                    {'type': 'call', 'position': 'short', 'strike': K, 'price': call_price},
                    {'type': 'put', 'position': 'short', 'strike': K, 'price': put_price}
                ],
                'strike': K,
                'expiry': T,
                'quantity': position_size,
                'credit': total_premium,
                'max_profit': total_premium,
                'max_loss': np.inf,
                'breakeven_upper': K + (call_price + put_price),
                'breakeven_lower': K - (call_price + put_price)
            }
    
    def calculate_strategy_pnl(self, strategy: Dict[str, Any], current_price: float) -> float:
        """Calculate current P&L for a strategy"""
        
        strategy_type = strategy['strategy']
        
        if strategy_type == 'long_call':
            intrinsic_value = max(0, current_price - strategy['strike'])
            return (intrinsic_value - strategy['premium']) * strategy['quantity']
        
        elif strategy_type == 'long_put':
            intrinsic_value = max(0, strategy['strike'] - current_price)
            return (intrinsic_value - strategy['premium']) * strategy['quantity']
        
        elif strategy_type == 'covered_call':
            stock_pnl = (current_price - strategy['components'][0]['price']) * strategy['quantity']
            call_intrinsic = max(0, current_price - strategy['strike'])
            call_pnl = (strategy['components'][1]['price'] - call_intrinsic) * strategy['quantity']
            return stock_pnl + call_pnl
        
        elif strategy_type == 'protective_put':
            stock_pnl = (current_price - strategy['components'][0]['price']) * strategy['quantity']
            put_intrinsic = max(0, strategy['strike'] - current_price)
            put_pnl = (put_intrinsic - strategy['components'][1]['price']) * strategy['quantity']
            return stock_pnl + put_pnl
        
        elif strategy_type == 'iron_condor':
            # Calculate payoff for each leg
            total_pnl = strategy['net_credit']
            
            for component in strategy['components']:
                if component['type'] == 'put':
                    intrinsic = max(0, component['strike'] - current_price)
                else:  # call
                    intrinsic = max(0, current_price - component['strike'])
                
                if component['position'] == 'long':
                    total_pnl += (intrinsic - component['price']) * strategy['quantity']
                else:  # short
                    total_pnl += (component['price'] - intrinsic) * strategy['quantity']
            
            return total_pnl
        
        elif strategy_type in ['long_straddle', 'short_straddle']:
            call_intrinsic = max(0, current_price - strategy['strike'])
            put_intrinsic = max(0, strategy['strike'] - current_price)
            
            if strategy_type == 'long_straddle':
                call_pnl = call_intrinsic - strategy['components'][0]['price']
                put_pnl = put_intrinsic - strategy['components'][1]['price']
                return (call_pnl + put_pnl) * strategy['quantity']
            else:  # short_straddle
                call_pnl = strategy['components'][0]['price'] - call_intrinsic
                put_pnl = strategy['components'][1]['price'] - put_intrinsic
                return (call_pnl + put_pnl) * strategy['quantity']
        
        return 0

class PortfolioBacktester:
    """Comprehensive portfolio backtesting engine"""
    
    def __init__(self):
        self.strategy_engine = OptionTradingStrategy()
        self.market_provider = MarketDataProvider()
    
    def run_backtest(self, universe_data: Dict[str, pd.DataFrame], 
                    strategy_config: Dict[str, Any],
                    start_date: str, end_date: str) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        
        try:
            # Initialize backtest
            initial_capital = strategy_config['initial_capital']
            position_size_pct = strategy_config['position_size_pct']
            commission = strategy_config.get('commission', 1.0)
            
            # Create date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get trading days
            sample_data = next(iter(universe_data.values()))
            trading_days = sample_data.loc[start_date:end_date].index
            
            if len(trading_days) == 0:
                st.error("No trading days found in the specified date range")
                return {}
            
            # Initialize tracking variables
            portfolio_history = []
            trades = []
            positions = []
            current_capital = initial_capital
            
            # Track performance metrics
            daily_returns = []
            portfolio_values = [initial_capital]
            
            # Benchmark (SPY if available)
            benchmark_data = universe_data.get('SPY')
            if benchmark_data is not None:
                benchmark_start = benchmark_data.loc[start_date:end_date]['Close'].iloc[0]
                benchmark_values = []
            
            # Main backtesting loop
            for i, date in enumerate(trading_days):
                date_str = date.strftime('%Y-%m-%d')
                
                # Get current prices for universe
                current_prices = {}
                for symbol, data in universe_data.items():
                    if date in data.index:
                        current_prices[symbol] = data.loc[date, 'Close']
                
                if not current_prices:
                    continue
                
                # Calculate current portfolio value
                portfolio_value = current_capital
                
                # Update existing positions
                for position in positions:
                    if position['symbol'] in current_prices:
                        current_price = current_prices[position['symbol']]
                        position_pnl = self.strategy_engine.calculate_strategy_pnl(
                            position['strategy'], current_price
                        )
                        portfolio_value += position_pnl
                
                # Trading logic based on strategy type
                if i > 0:  # Skip first day for signal generation
                    new_trades = self._generate_trading_signals(
                        strategy_config, current_prices, universe_data, date, current_capital
                    )
                    
                    for trade in new_trades:
                        trades.append(trade)
                        
                        # Update positions
                        if trade['action'] == 'open':
                            positions.append(trade)
                            current_capital -= trade['cost'] + commission
                        elif trade['action'] == 'close':
                            # Find and close matching position
                            for j, pos in enumerate(positions):
                                if pos['id'] == trade['position_id']:
                                    current_capital += trade['proceeds'] - commission
                                    positions.pop(j)
                                    break
                
                # Calculate daily return
                if i > 0:
                    daily_return = (portfolio_value - portfolio_values[-1]) / portfolio_values[-1]
                    daily_returns.append(daily_return)
                
                portfolio_values.append(portfolio_value)
                
                # Track benchmark
                if benchmark_data is not None and date in benchmark_data.index:
                    benchmark_current = benchmark_data.loc[date, 'Close']
                    benchmark_value = initial_capital * (benchmark_current / benchmark_start)
                    benchmark_values.append(benchmark_value)
                else:
                    benchmark_values.append(initial_capital)
                
                # Record portfolio history
                portfolio_history.append({
                    'date': date_str,
                    'portfolio_value': portfolio_value,
                    'cash': current_capital,
                    'positions_value': portfolio_value - current_capital,
                    'num_positions': len(positions),
                    'benchmark_value': benchmark_values[-1] if benchmark_values else initial_capital
                })
            
            # Calculate performance metrics
            if daily_returns:
                performance_metrics = self._calculate_performance_metrics(
                    daily_returns, portfolio_values, initial_capital
                )
            else:
                performance_metrics = {}
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(daily_returns, portfolio_values)
            
            # Rolling metrics
            rolling_metrics = self._calculate_rolling_metrics(daily_returns)
            
            return {
                'portfolio_history': portfolio_history,
                'trades': trades,
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics,
                'rolling_sharpe': rolling_metrics.get('rolling_sharpe', []),
                'daily_returns': daily_returns,
                'final_portfolio_value': portfolio_values[-1],
                'backtest_config': strategy_config,
                'universe': list(universe_data.keys()),
                'period': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
            return {}
    
    def _generate_trading_signals(self, strategy_config: Dict[str, Any], 
                                 current_prices: Dict[str, float],
                                 universe_data: Dict[str, pd.DataFrame],
                                 current_date: pd.Timestamp,
                                 available_capital: float) -> List[Dict[str, Any]]:
        """Generate trading signals based on strategy"""
        
        signals = []
        strategy_type = strategy_config['strategy_type']
        position_size_pct = strategy_config['position_size_pct']
        
        # Simple signal generation logic
        for symbol, price in current_prices.items():
            if symbol not in universe_data:
                continue
            
            symbol_data = universe_data[symbol]
            
            # Get recent data for signal generation
            recent_data = symbol_data.loc[:current_date].tail(20)
            if len(recent_data) < 10:
                continue
            
            # Calculate some technical indicators
            recent_returns = recent_data['Close'].pct_change().dropna()
            volatility = recent_returns.std() * np.sqrt(252)
            
            # Simple momentum signal
            momentum = (price / recent_data['Close'].iloc[-5]) - 1 if len(recent_data) >= 5 else 0
            
            # Generate strategy parameters
            risk_free_rate = 0.05  # Assume 5% risk-free rate
            time_to_expiry = strategy_config.get('days_to_expiration', 30) / 365
            
            # Position sizing
            position_value = available_capital * position_size_pct
            
            # Strategy-specific logic
            if strategy_type == "Long Call" and momentum > 0.02:  # Bullish signal
                moneyness = strategy_config.get('moneyness', 1.0)
                strike = price * moneyness
                
                strategy_def = self.strategy_engine.long_call_strategy(
                    price, strike, time_to_expiry, risk_free_rate, volatility, 1
                )
                
                if strategy_def['cost'] <= position_value:
                    signals.append({
                        'id': f"{symbol}_{current_date.strftime('%Y%m%d')}_{len(signals)}",
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'open',
                        'strategy': strategy_def,
                        'cost': strategy_def['cost'],
                        'signal_strength': abs(momentum)
                    })
            
            elif strategy_type == "Long Put" and momentum < -0.02:  # Bearish signal
                moneyness = strategy_config.get('moneyness', 1.0)
                strike = price * moneyness
                
                strategy_def = self.strategy_engine.long_put_strategy(
                    price, strike, time_to_expiry, risk_free_rate, volatility, 1
                )
                
                if strategy_def['cost'] <= position_value:
                    signals.append({
                        'id': f"{symbol}_{current_date.strftime('%Y%m%d')}_{len(signals)}",
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'open',
                        'strategy': strategy_def,
                        'cost': strategy_def['cost'],
                        'signal_strength': abs(momentum)
                    })
            
            # Add more strategy types as needed
        
        return signals[:3]  # Limit to 3 new positions per day
    
    def _calculate_performance_metrics(self, daily_returns: List[float], 
                                     portfolio_values: List[float],
                                     initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if not daily_returns or not portfolio_values:
            return {}
        
        returns_array = np.array(daily_returns)
        final_value = portfolio_values[-1]
        
        # Basic metrics
        total_return = (final_value / initial_capital) - 1
        annualized_return = (final_value / initial_capital) ** (252 / len(daily_returns)) - 1
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = RiskMetrics.sharpe_ratio(pd.Series(returns_array))
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_days = np.sum(returns_array > 0)
        win_rate = positive_days / len(returns_array)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'best_day': np.max(returns_array),
            'worst_day': np.min(returns_array),
            'positive_days': positive_days,
            'negative_days': len(returns_array) - positive_days
        }
    
    def _calculate_risk_metrics(self, daily_returns: List[float], 
                               portfolio_values: List[float]) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        if not daily_returns:
            return {}
        
        returns_series = pd.Series(daily_returns)
        
        # VaR calculations
        var_95 = RiskMetrics.value_at_risk(returns_series, 0.05)
        cvar_95 = RiskMetrics.conditional_var(returns_series, 0.05)
        
        # Convert to dollar amounts (approximate)
        portfolio_value = portfolio_values[-1] if portfolio_values else 100000
        var_95_dollar = var_95 * portfolio_value
        cvar_95_dollar = cvar_95 * portfolio_value
        
        return {
            'var_95': var_95_dollar,
            'cvar_95': cvar_95_dollar,
            'volatility': returns_series.std() * np.sqrt(252),
            'downside_deviation': returns_series[returns_series < 0].std() * np.sqrt(252) if len(returns_series[returns_series < 0]) > 0 else 0,
            'calmar_ratio': self._calculate_performance_metrics(daily_returns, portfolio_values, portfolio_values[0] if portfolio_values else 100000).get('calmar_ratio', 0)
        }
    
    def _calculate_rolling_metrics(self, daily_returns: List[float], 
                                  window: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate rolling performance metrics"""
        
        if len(daily_returns) < window:
            return {}
        
        returns_series = pd.Series(daily_returns)
        rolling_sharpe = []
        
        for i in range(window, len(returns_series) + 1):
            window_returns = returns_series.iloc[i-window:i]
            if len(window_returns) == window:
                sharpe = RiskMetrics.sharpe_ratio(window_returns)
                rolling_sharpe.append({
                    'date': f"Day {i}",  # In real implementation, use actual dates
                    'sharpe_ratio': sharpe
                })
        
        return {
            'rolling_sharpe': rolling_sharpe
        }

class StrategyOptimizer:
    """Optimize strategy parameters using historical data"""
    
    def __init__(self):
        self.backtester = PortfolioBacktester()
    
    def optimize_strategy_parameters(self, universe_data: Dict[str, pd.DataFrame],
                                   strategy_type: str, parameter_ranges: Dict[str, Tuple[float, float]],
                                   optimization_metric: str = 'sharpe_ratio',
                                   start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search"""
        
        try:
            # Generate parameter combinations
            param_combinations = self._generate_parameter_grid(parameter_ranges)
            
            best_params = None
            best_score = -np.inf
            optimization_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, params in enumerate(param_combinations):
                status_text.text(f"Testing parameter combination {i+1}/{len(param_combinations)}")
                progress_bar.progress((i + 1) / len(param_combinations))
                
                # Create strategy config
                strategy_config = {
                    'strategy_type': strategy_type,
                    'initial_capital': 100000,
                    'position_size_pct': 0.1,
                    'commission': 1.0,
                    **params
                }
                
                # Run backtest
                backtest_results = self.backtester.run_backtest(
                    universe_data, strategy_config, start_date, end_date
                )
                
                if backtest_results and 'performance_metrics' in backtest_results:
                    metrics = backtest_results['performance_metrics']
                    score = metrics.get(optimization_metric, -np.inf)
                    
                    optimization_results.append({
                        'parameters': params,
                        'score': score,
                        'metrics': metrics
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
            
            progress_bar.empty()
            status_text.empty()
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'optimization_metric': optimization_metric,
                'all_results': optimization_results,
                'parameter_ranges': parameter_ranges
            }
            
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            return {}
    
    def _generate_parameter_grid(self, parameter_ranges: Dict[str, Tuple[float, float]],
                                grid_size: int = 5) -> List[Dict[str, float]]:
        """Generate grid of parameter combinations"""
        
        param_grids = {}
        for param, (min_val, max_val) in parameter_ranges.items():
            param_grids[param] = np.linspace(min_val, max_val, grid_size)
        
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_combinations = []
        
        import itertools
        for combination in itertools.product(*param_grids.values()):
            param_dict = dict(zip(param_names, combination))
            param_combinations.append(param_dict)
        
        return param_combinations[:20]  # Limit to 20 combinations for performance

class RiskManagement:
    """Risk management and position sizing for strategies"""
    
    def __init__(self):
        self.max_portfolio_risk = 0.02  # 2% daily VaR limit
        self.max_position_size = 0.1    # 10% max position size
        self.correlation_limit = 0.7    # Max correlation between positions
    
    def calculate_position_size(self, strategy_config: Dict[str, Any],
                               portfolio_data: Dict[str, Any],
                               market_conditions: Dict[str, float]) -> float:
        """Calculate optimal position size based on risk management rules"""
        
        try:
            # Kelly criterion approximation
            win_rate = portfolio_data.get('historical_win_rate', 0.5)
            avg_win = portfolio_data.get('avg_winning_trade', 0.02)
            avg_loss = abs(portfolio_data.get('avg_losing_trade', -0.015))
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.1
            
            # Volatility-based sizing
            current_vol = market_conditions.get('volatility', 0.2)
            target_vol = 0.15
            vol_adjustment = target_vol / current_vol if current_vol > 0 else 1.0
            
            # Risk parity adjustment
            portfolio_vol = portfolio_data.get('portfolio_volatility', 0.15)
            risk_budget = self.max_portfolio_risk / portfolio_vol if portfolio_vol > 0 else 0.1
            
            # Combine sizing methods
            position_size = min(
                kelly_fraction * 0.5,  # Conservative Kelly
                vol_adjustment * 0.1,   # Volatility target
                risk_budget,           # Risk budget
                self.max_position_size # Hard limit
            )
            
            return max(0.01, position_size)  # Minimum 1% position
            
        except Exception as e:
            st.warning(f"Position sizing calculation failed: {str(e)}")
            return 0.05  # Default 5%
    
    def check_risk_limits(self, current_positions: List[Dict[str, Any]],
                         new_trade: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if new trade violates risk limits"""
        
        # Check position concentration
        total_exposure = sum([pos.get('exposure', 0) for pos in current_positions])
        if new_trade.get('exposure', 0) + total_exposure > 0.8:  # 80% max exposure
            return False, "Portfolio exposure limit exceeded"
        
        # Check correlation limits
        new_symbol = new_trade.get('symbol', '')
        for pos in current_positions:
            if pos.get('symbol', '') == new_symbol:
                return False, f"Already have position in {new_symbol}"
        
        # Check sector concentration
        new_sector = new_trade.get('sector', 'Unknown')
        sector_exposure = sum([pos.get('exposure', 0) for pos in current_positions 
                              if pos.get('sector', '') == new_sector])
        
        if sector_exposure + new_trade.get('exposure', 0) > 0.3:  # 30% sector limit
            return False, f"Sector concentration limit exceeded for {new_sector}"
        
        return True, "Risk checks passed"
    
    def calculate_portfolio_var(self, positions: List[Dict[str, Any]],
                               confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate portfolio Value at Risk"""
        
        try:
            if not positions:
                return {'var': 0, 'cvar': 0, 'individual_vars': {}}
            
            # Simulate portfolio returns (simplified)
            portfolio_value = sum([pos.get('market_value', 0) for pos in positions])
            
            if portfolio_value == 0:
                return {'var': 0, 'cvar': 0, 'individual_vars': {}}
            
            # Monte Carlo simulation for portfolio VaR
            n_simulations = 1000
            portfolio_returns = []
            
            for _ in range(n_simulations):
                daily_return = 0
                for pos in positions:
                    # Simulate individual position return
                    pos_vol = pos.get('volatility', 0.2)
                    pos_return = np.random.normal(0, pos_vol / np.sqrt(252))
                    pos_weight = pos.get('market_value', 0) / portfolio_value
                    daily_return += pos_return * pos_weight
                
                portfolio_returns.append(daily_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            # Calculate VaR and CVaR
            var = np.percentile(portfolio_returns, confidence_level * 100) * portfolio_value
            cvar = np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, confidence_level * 100)]) * portfolio_value
            
            # Individual position VaRs
            individual_vars = {}
            for pos in positions:
                pos_vol = pos.get('volatility', 0.2)
                pos_value = pos.get('market_value', 0)
                pos_var = np.percentile(np.random.normal(0, pos_vol / np.sqrt(252), 1000), confidence_level * 100) * pos_value
                individual_vars[pos.get('symbol', 'unknown')] = pos_var
            
            return {
                'var': abs(var),
                'cvar': abs(cvar),
                'individual_vars': individual_vars,
                'portfolio_value': portfolio_value
            }
            
        except Exception as e:
            st.error(f"VaR calculation failed: {str(e)}")
            return {'var': 0, 'cvar': 0, 'individual_vars': {}}
