"""
AI-Enhanced Financial Models
Advanced machine learning models for quantitative finance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from scipy.optimize import minimize
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime, timedelta

class QuantumInspiredOptimizer:
    """Quantum-inspired optimization for portfolio allocation"""
    
    def __init__(self):
        self.name = "Quantum-Inspired Portfolio Optimizer"
    
    def quantum_portfolio_optimization(self, returns: pd.DataFrame, 
                                     risk_tolerance: float = 0.5,
                                     quantum_iterations: int = 1000) -> Dict:
        """
        Use quantum-inspired algorithms for portfolio optimization
        
        Args:
            returns: Historical returns matrix
            risk_tolerance: Risk tolerance (0 = risk-averse, 1 = risk-seeking)
            quantum_iterations: Number of quantum-inspired iterations
        """
        try:
            n_assets = len(returns.columns)
            
            # Calculate expected returns and covariance matrix
            mu = returns.mean() * 252  # Annualized returns
            cov = returns.cov() * 252  # Annualized covariance
            
            # Quantum-inspired superposition initialization
            weights = self._initialize_quantum_superposition(n_assets)
            
            # Quantum annealing-inspired optimization
            best_weights = weights.copy()
            best_score = -np.inf
            
            # Simulated quantum annealing
            temperature = 1.0
            cooling_rate = 0.995
            
            for iteration in range(quantum_iterations):
                # Quantum measurement (collapse superposition)
                trial_weights = self._quantum_measurement(weights, temperature)
                trial_weights = self._normalize_weights(trial_weights)
                
                # Calculate utility score
                portfolio_return = np.dot(trial_weights, mu)
                portfolio_risk = np.sqrt(np.dot(trial_weights, np.dot(cov, trial_weights)))
                
                # Quantum utility function with risk tolerance
                utility = portfolio_return - (1 - risk_tolerance) * portfolio_risk**2
                
                # Quantum acceptance probability
                if utility > best_score or np.random.random() < np.exp((utility - best_score) / temperature):
                    best_weights = trial_weights.copy()
                    best_score = utility
                    
                    # Update quantum state
                    weights = self._update_quantum_state(weights, trial_weights, temperature)
                
                # Cool down
                temperature *= cooling_rate
            
            # Calculate final portfolio metrics
            final_return = np.dot(best_weights, mu)
            final_risk = np.sqrt(np.dot(best_weights, np.dot(cov, best_weights)))
            sharpe_ratio = final_return / final_risk if final_risk > 0 else 0
            
            return {
                'optimal_weights': best_weights,
                'expected_return': final_return,
                'expected_risk': final_risk,
                'sharpe_ratio': sharpe_ratio,
                'utility_score': best_score,
                'quantum_iterations': quantum_iterations,
                'assets': list(returns.columns)
            }
            
        except Exception as e:
            st.error(f"Quantum optimization error: {str(e)}")
            # Fallback to equal weights
            n = len(returns.columns)
            return {
                'optimal_weights': np.ones(n) / n,
                'expected_return': 0,
                'expected_risk': 0,
                'sharpe_ratio': 0,
                'error': str(e)
            }
    
    def _initialize_quantum_superposition(self, n_assets: int) -> np.ndarray:
        """Initialize weights in quantum superposition"""
        # Complex amplitudes for quantum superposition
        real_part = np.random.normal(0, 1, n_assets)
        imag_part = np.random.normal(0, 1, n_assets)
        
        # Convert to probability amplitudes
        amplitudes = np.sqrt(real_part**2 + imag_part**2)
        return amplitudes / np.sum(amplitudes)
    
    def _quantum_measurement(self, weights: np.ndarray, temperature: float) -> np.ndarray:
        """Perform quantum measurement with temperature-dependent noise"""
        noise = np.random.normal(0, temperature * 0.1, len(weights))
        return np.abs(weights + noise)
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1"""
        weights = np.abs(weights)  # Ensure positive
        return weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)
    
    def _update_quantum_state(self, current: np.ndarray, trial: np.ndarray, temperature: float) -> np.ndarray:
        """Update quantum state based on successful trial"""
        learning_rate = 0.1 * temperature
        return current + learning_rate * (trial - current)


class ReinforcementLearningTrader:
    """Reinforcement learning for dynamic trading strategies"""
    
    def __init__(self):
        self.name = "RL Trading Agent"
        self.state_dim = 20  # Feature dimensions
        self.action_dim = 3  # Buy, Hold, Sell
        self.q_table = {}
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.01
        self.discount_factor = 0.95
    
    def train_rl_agent(self, price_data: pd.DataFrame, 
                      lookback_window: int = 20,
                      episodes: int = 1000) -> Dict:
        """
        Train reinforcement learning trading agent
        
        Args:
            price_data: OHLCV price data
            lookback_window: Number of periods for state calculation
            episodes: Number of training episodes
        """
        try:
            # Prepare features for RL state
            features = self._prepare_rl_features(price_data, lookback_window)
            
            # Training metrics
            episode_rewards = []
            episode_trades = []
            
            # Training loop
            for episode in range(episodes):
                total_reward = 0
                position = 0  # -1: short, 0: neutral, 1: long
                cash = 10000  # Starting cash
                trades = 0
                
                for i in range(lookback_window, len(features) - 1):
                    # Current state
                    state = self._encode_state(features.iloc[i], position)
                    
                    # Choose action (epsilon-greedy)
                    action = self._choose_action(state)
                    
                    # Execute action
                    current_price = price_data['Close'].iloc[i]
                    next_price = price_data['Close'].iloc[i + 1]
                    
                    reward, new_position, trade_occurred = self._execute_action(
                        action, position, current_price, next_price, cash
                    )
                    
                    if trade_occurred:
                        trades += 1
                    
                    # Next state
                    next_state = self._encode_state(features.iloc[i + 1], new_position)
                    
                    # Update Q-table
                    self._update_q_table(state, action, reward, next_state)
                    
                    total_reward += reward
                    position = new_position
                
                episode_rewards.append(total_reward)
                episode_trades.append(trades)
                
                # Decay epsilon
                self.epsilon = max(0.01, self.epsilon * 0.995)
            
            # Calculate performance metrics
            avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
            win_rate = len([r for r in episode_rewards[-100:] if r > 0]) / 100
            
            return {
                'trained_episodes': episodes,
                'average_reward': avg_reward,
                'win_rate': win_rate,
                'total_trades': np.mean(episode_trades[-100:]),
                'q_table_size': len(self.q_table),
                'final_epsilon': self.epsilon,
                'episode_rewards': episode_rewards,
                'convergence_score': self._calculate_convergence(episode_rewards)
            }
            
        except Exception as e:
            st.error(f"RL training error: {str(e)}")
            return {'trained_episodes': 0, 'error': str(e)}
    
    def _prepare_rl_features(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """Prepare features for RL state representation"""
        features = pd.DataFrame(index=data.index)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['macd'] = self._calculate_macd(data['Close'])
        features['bb_position'] = self._bollinger_position(data['Close'])
        features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window).mean()
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(window).std()
        features['momentum'] = data['Close'] / data['Close'].shift(window) - 1
        
        # Trend features
        features['sma_short'] = data['Close'].rolling(5).mean()
        features['sma_long'] = data['Close'].rolling(20).mean()
        features['trend'] = (features['sma_short'] > features['sma_long']).astype(int)
        
        return features.fillna(0)
    
    def _encode_state(self, features: pd.Series, position: int) -> str:
        """Encode state for Q-table lookup"""
        # Discretize continuous features
        state_vector = []
        
        for feature in features:
            if feature < -2:
                state_vector.append('low')
            elif feature < 2:
                state_vector.append('mid')
            else:
                state_vector.append('high')
        
        state_vector.append(f"pos_{position}")
        return "_".join(state_vector)
    
    def _choose_action(self, state: str) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)  # Explore
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        
        return np.argmax(self.q_table[state])  # Exploit
    
    def _execute_action(self, action: int, position: int, 
                       current_price: float, next_price: float, cash: float) -> Tuple[float, int, bool]:
        """Execute trading action and calculate reward"""
        reward = 0
        new_position = position
        trade_occurred = False
        
        price_change = (next_price - current_price) / current_price
        
        if action == 0 and position != 1:  # Buy
            new_position = 1
            trade_occurred = True
            reward = price_change * 100  # Scale reward
        elif action == 2 and position != -1:  # Sell
            new_position = -1
            trade_occurred = True
            reward = -price_change * 100  # Profit from price decline
        elif action == 1:  # Hold
            reward = position * price_change * 100
        
        # Penalty for excessive trading
        if trade_occurred:
            reward -= 0.1  # Transaction cost
        
        return reward, new_position, trade_occurred
    
    def _update_q_table(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-table using Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_dim)
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def _bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position relative to Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (prices - lower) / (upper - lower)
    
    def _calculate_convergence(self, rewards: List[float]) -> float:
        """Calculate convergence score of RL training"""
        if len(rewards) < 100:
            return 0
        
        recent_rewards = rewards[-100:]
        early_rewards = rewards[:100]
        
        return np.mean(recent_rewards) - np.mean(early_rewards)


class TransformerPricePredictor:
    """Transformer-based price prediction model"""
    
    def __init__(self):
        self.name = "Transformer Price Predictor"
        self.sequence_length = 60
        self.feature_dim = 10
    
    def create_transformer_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create feature sequences for transformer model"""
        try:
            features = []
            
            # Price-based features
            features.append(data['Close'].pct_change().fillna(0))
            features.append(data['High'].pct_change().fillna(0))
            features.append(data['Low'].pct_change().fillna(0))
            features.append((data['Volume'] / data['Volume'].rolling(20).mean()).fillna(1))
            
            # Technical indicators
            features.append(self._rsi(data['Close']))
            features.append(self._macd(data['Close']))
            features.append(self._bollinger_position(data['Close']))
            
            # Volatility features
            returns = data['Close'].pct_change()
            features.append(returns.rolling(20).std().fillna(0))
            features.append(returns.rolling(5).skew().fillna(0))
            features.append(returns.rolling(5).kurt().fillna(0))
            
            # Stack features
            feature_matrix = np.column_stack(features)
            
            # Create sequences
            sequences = []
            for i in range(self.sequence_length, len(feature_matrix)):
                sequences.append(feature_matrix[i-self.sequence_length:i])
            
            return np.array(sequences)
            
        except Exception as e:
            st.error(f"Feature creation error: {str(e)}")
            return np.array([])
    
    def attention_mechanism_simulation(self, sequences: np.ndarray) -> Dict:
        """Simulate transformer attention mechanism for price prediction"""
        try:
            if len(sequences) == 0:
                return {'predictions': [], 'attention_weights': []}
            
            # Simplified attention mechanism
            predictions = []
            attention_weights = []
            
            for seq in sequences[-100:]:  # Last 100 sequences
                # Calculate attention scores (simplified)
                # In real transformer: Q, K, V matrices and multi-head attention
                
                # Recent time steps get higher attention
                time_weights = np.exp(np.arange(len(seq)) * 0.1)
                time_weights = time_weights / np.sum(time_weights)
                
                # Feature importance weights
                feature_weights = np.array([0.3, 0.2, 0.2, 0.1, 0.15, 0.05, 0.1, 0.1, 0.05, 0.05])
                
                # Combine temporal and feature attention
                attention_matrix = np.outer(time_weights, feature_weights)
                
                # Weighted prediction
                weighted_features = np.sum(seq * attention_matrix, axis=0)
                prediction = np.dot(weighted_features, feature_weights)
                
                predictions.append(prediction)
                attention_weights.append(attention_matrix)
            
            return {
                'predictions': predictions,
                'attention_weights': attention_weights,
                'sequence_length': self.sequence_length,
                'feature_importance': feature_weights.tolist(),
                'temporal_importance': time_weights.tolist()
            }
            
        except Exception as e:
            st.error(f"Attention mechanism error: {str(e)}")
            return {'predictions': [], 'error': str(e)}
    
    def _rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).fillna(50) / 100  # Normalize to 0-1
    
    def _macd(self, prices: pd.Series) -> pd.Series:
        """Calculate normalized MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        return (macd / prices).fillna(0)  # Normalize by price
    
    def _bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Bollinger Band position"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return ((prices - lower) / (upper - lower)).fillna(0.5)


class AutoMLFinancialModels:
    """Automated machine learning for financial modeling"""
    
    def __init__(self):
        self.name = "AutoML Financial Models"
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
    
    def auto_model_selection(self, X: pd.DataFrame, y: pd.Series, 
                           problem_type: str = 'regression') -> Dict:
        """
        Automatically select and tune the best model
        
        Args:
            X: Feature matrix
            y: Target variable
            problem_type: 'regression' or 'classification'
        """
        try:
            # Prepare data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Model candidates
            if problem_type == 'regression':
                models = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
                }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            results = {}
            for name, model in models.items():
                try:
                    # Cross-validation scores
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                              scoring='neg_mean_squared_error')
                    
                    # Train full model
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    
                    # Feature importance (if available)
                    importance = None
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        importance = np.abs(model.coef_)
                    
                    results[name] = {
                        'model': model,
                        'cv_scores': cv_scores,
                        'cv_mean': np.mean(cv_scores),
                        'cv_std': np.std(cv_scores),
                        'mse': mse,
                        'r2': r2,
                        'mae': mae,
                        'feature_importance': importance
                    }
                    
                except Exception as e:
                    st.warning(f"Error training {name}: {str(e)}")
                    continue
            
            # Select best model
            if results:
                best_name = max(results.keys(), key=lambda x: results[x]['r2'])
                self.best_model = results[best_name]['model']
                self.models = results
                
                # Feature importance analysis
                if results[best_name]['feature_importance'] is not None:
                    importance_dict = dict(zip(X.columns, results[best_name]['feature_importance']))
                    self.feature_importance = dict(sorted(importance_dict.items(), 
                                                        key=lambda x: abs(x[1]), reverse=True))
                
                return {
                    'best_model': best_name,
                    'best_score': results[best_name]['r2'],
                    'all_results': results,
                    'feature_importance': self.feature_importance,
                    'model_count': len(results)
                }
            else:
                return {'error': 'No models trained successfully'}
                
        except Exception as e:
            st.error(f"AutoML error: {str(e)}")
            return {'error': str(e)}
    
    def hyperparameter_optimization(self, model_name: str, X: pd.DataFrame, 
                                  y: pd.Series, n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using random search
        """
        try:
            if model_name not in self.models:
                return {'error': f'Model {model_name} not found'}
            
            base_model = self.models[model_name]['model']
            
            # Define hyperparameter spaces
            param_spaces = {
                'random_forest': {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
            
            if model_name not in param_spaces:
                return {'error': f'No hyperparameter space defined for {model_name}'}
            
            param_space = param_spaces[model_name]
            
            # Random search
            best_score = -np.inf
            best_params = {}
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tscv = TimeSeriesSplit(n_splits=3)
            
            for _ in range(n_trials):
                # Random parameter selection
                params = {}
                for param, values in param_space.items():
                    params[param] = np.random.choice(values)
                
                try:
                    # Create and evaluate model
                    if model_name == 'random_forest':
                        model = RandomForestRegressor(**params, random_state=42)
                    elif model_name == 'xgboost':
                        model = xgb.XGBRegressor(**params, random_state=42)
                    
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                              scoring='neg_mean_squared_error')
                    score = np.mean(cv_scores)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception:
                    continue
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'trials_completed': n_trials,
                'improvement': best_score - self.models[model_name]['cv_mean']
            }
            
        except Exception as e:
            st.error(f"Hyperparameter optimization error: {str(e)}")
            return {'error': str(e)}


# Initialize AI-enhanced models
@st.cache_resource
def get_quantum_optimizer():
    return QuantumInspiredOptimizer()

@st.cache_resource
def get_rl_trader():
    return ReinforcementLearningTrader()

@st.cache_resource
def get_transformer_predictor():
    return TransformerPricePredictor()

@st.cache_resource
def get_automl_models():
    return AutoMLFinancialModels()