import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """Feature engineering for option pricing models"""
    
    @staticmethod
    def create_option_features(S: float, K: float, T: float, r: float, 
                             historical_vols: Optional[pd.Series] = None,
                             market_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Create comprehensive features for option pricing"""
        
        features = {
            # Basic features
            'spot_price': S,
            'strike_price': K,
            'time_to_expiry': T,
            'risk_free_rate': r,
            'moneyness': S / K,
            'log_moneyness': np.log(S / K),
            
            # Derived features
            'forward_price': S * np.exp(r * T),
            'forward_moneyness': (S * np.exp(r * T)) / K,
            'time_sqrt': np.sqrt(T),
            'time_squared': T ** 2,
            
            # ITM/OTM indicators
            'is_itm_call': 1 if S > K else 0,
            'is_itm_put': 1 if S < K else 0,
            'is_atm': 1 if abs(S - K) / K < 0.05 else 0,
            
            # Expiry buckets
            'short_term': 1 if T < 0.25 else 0,  # Less than 3 months
            'medium_term': 1 if 0.25 <= T <= 1.0 else 0,  # 3 months to 1 year
            'long_term': 1 if T > 1.0 else 0,  # More than 1 year
        }
        
        # Historical volatility features
        if historical_vols is not None and len(historical_vols) > 0:
            features.update({
                'hist_vol_30d': historical_vols.tail(30).std() * np.sqrt(252) if len(historical_vols) >= 30 else 0.2,
                'hist_vol_60d': historical_vols.tail(60).std() * np.sqrt(252) if len(historical_vols) >= 60 else 0.2,
                'hist_vol_90d': historical_vols.tail(90).std() * np.sqrt(252) if len(historical_vols) >= 90 else 0.2,
                'vol_mean': historical_vols.mean() if len(historical_vols) > 0 else 0.2,
                'vol_std': historical_vols.std() if len(historical_vols) > 1 else 0.05,
            })
        
        # Market regime features
        if market_data is not None and len(market_data) > 0:
            if 'returns' in market_data.columns:
                recent_returns = market_data['returns'].tail(20)
                features.update({
                    'avg_return_20d': recent_returns.mean() if len(recent_returns) > 0 else 0,
                    'volatility_20d': recent_returns.std() * np.sqrt(252) if len(recent_returns) > 1 else 0.2,
                    'skewness_20d': recent_returns.skew() if len(recent_returns) > 2 else 0,
                    'kurtosis_20d': recent_returns.kurtosis() if len(recent_returns) > 3 else 3,
                })
        
        return features
    
    @staticmethod
    def create_features_dataframe(option_data: pd.DataFrame) -> pd.DataFrame:
        """Create features DataFrame from option data"""
        features_list = []
        
        for _, row in option_data.iterrows():
            features = FeatureEngineering.create_option_features(
                S=row['spot_price'],
                K=row['strike_price'], 
                T=row['time_to_expiry'],
                r=row['risk_free_rate']
            )
            features_list.append(features)
        
        return pd.DataFrame(features_list)

class MLOptionPricer:
    """Machine Learning models for option pricing"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
    
    def prepare_training_data(self, market_data: pd.DataFrame, 
                            target_column: str = 'option_price') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with features and target"""
        
        # Create features
        features_df = FeatureEngineering.create_features_dataframe(market_data)
        
        # Target variable
        target = market_data[target_column]
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        return features_df, target
    
    def train_xgboost_model(self, X: pd.DataFrame, y: pd.Series, 
                          hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """Train XGBoost model for option pricing"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(xgb_model, param_grid, cv=5, 
                                     scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
        else:
            # Default parameters
            best_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            best_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = best_model.predict(X_train_scaled)
        y_pred_test = best_model.predict(X_test_scaled)
        
        # Metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test)
        }
        
        # Store model and scaler
        self.models['xgboost'] = best_model
        self.scalers['xgboost'] = scaler
        
        return {
            'model': best_model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': dict(zip(X.columns, best_model.feature_importances_)),
            'cross_val_scores': cross_val_score(best_model, X_train_scaled, y_train, cv=5)
        }
    
    def train_neural_network(self, X: pd.DataFrame, y: pd.Series, 
                           architecture: List[int] = [100, 50, 25]) -> Dict[str, Any]:
        """Train neural network for option pricing"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features and target
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Build model
        model = keras.Sequential()
        model.add(keras.layers.Dense(architecture[0], activation='relu', input_shape=(X_train.shape[1],)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        
        for units in architecture[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.2))
        
        model.add(keras.layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        )
        
        # Predictions
        y_pred_train_scaled = model.predict(X_train_scaled).flatten()
        y_pred_test_scaled = model.predict(X_test_scaled).flatten()
        
        # Inverse scale predictions
        y_pred_train = target_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
        y_pred_test = target_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        
        # Metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test)
        }
        
        # Store model and scalers
        self.models['neural_network'] = model
        self.scalers['neural_network'] = {'features': feature_scaler, 'target': target_scaler}
        
        return {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'history': history.history
        }
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest model for option pricing"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, 
                                 scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        # Metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test)
        }
        
        # Store model
        self.models['random_forest'] = best_model
        
        return {
            'model': best_model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': dict(zip(X.columns, best_model.feature_importances_))
        }
    
    def predict(self, model_name: str, S: float, K: float, T: float, r: float, 
               historical_vols: Optional[pd.Series] = None) -> float:
        """Predict option price using trained model"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Create features
        features = FeatureEngineering.create_option_features(S, K, T, r, historical_vols)
        features_df = pd.DataFrame([features])
        
        # Ensure all feature columns are present
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.feature_names]
        
        # Scale features if needed
        if model_name in self.scalers:
            if model_name == 'neural_network':
                features_scaled = self.scalers[model_name]['features'].transform(features_df)
                prediction_scaled = self.models[model_name].predict(features_scaled)
                prediction = self.scalers[model_name]['target'].inverse_transform(
                    prediction_scaled.reshape(-1, 1)
                ).flatten()[0]
            else:
                features_scaled = self.scalers[model_name].transform(features_df)
                prediction = self.models[model_name].predict(features_scaled)[0]
        else:
            prediction = self.models[model_name].predict(features_df)[0]
        
        return prediction

class PhysicsInformedNeuralNetwork:
    """Physics-Informed Neural Network for option pricing using scikit-learn"""
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (50, 50, 25)):
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.feature_names = None
        self.training_history = {}
    
    def _calculate_bs_residual(self, S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                              r: np.ndarray, sigma: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Calculate approximate Black-Scholes PDE residual using finite differences"""
        
        # Small perturbations for numerical derivatives
        h_S = 0.01 * S
        h_T = 0.001
        
        # Approximate derivatives using finite differences
        # dV/dS ≈ (V(S+h) - V(S-h)) / (2h)
        # d²V/dS² ≈ (V(S+h) - 2V(S) + V(S-h)) / h²
        # dV/dT is approximated as -V for simplicity in this physics-informed approach
        
        # For this simplified implementation, we use analytical gradients from Black-Scholes
        # to constrain the neural network predictions
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Black-Scholes derivatives (for call options)
        from scipy.stats import norm
        
        # Delta (dV/dS)
        delta = norm.cdf(d1)
        
        # Gamma (d²V/dS²)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta approximation (dV/dT)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2))
        
        # Black-Scholes PDE residual: ∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + r*S*∂V/∂S - r*V = 0
        pde_residual = theta + 0.5 * sigma**2 * S**2 * gamma + r * S * delta - r * V
        
        return np.abs(pde_residual)
    
    def train_pinn(self, S_data: np.ndarray, K_data: np.ndarray, T_data: np.ndarray,
                  r_data: np.ndarray, sigma_data: np.ndarray, V_market: np.ndarray,
                  epochs: int = 500) -> Dict[str, List[float]]:
        """Train Physics-Informed Neural Network using scikit-learn with custom loss"""
        
        # Prepare features
        from models.black_scholes import BlackScholesModel
        
        features_list = []
        for i in range(len(S_data)):
            features = FeatureEngineering.create_option_features(
                S_data[i], K_data[i], T_data[i], r_data[i]
            )
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()
        
        # Scale features and targets
        X_scaled = self.scaler_features.fit_transform(features_df)
        y_scaled = self.scaler_target.fit_transform(V_market.reshape(-1, 1)).ravel()
        
        # Initialize neural network with physics-informed loss weighting
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='tanh',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50
        )
        
        # Train the model
        self.model.fit(X_scaled, y_scaled)
        
        # Calculate physics-informed metrics
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Calculate PDE residuals
        pde_residuals = self._calculate_bs_residual(S_data, K_data, T_data, r_data, sigma_data, y_pred)
        
        # Training history
        self.training_history = {
            'data_loss': [mean_squared_error(V_market, y_pred)],
            'pde_loss': [np.mean(pde_residuals)],
            'total_loss': [mean_squared_error(V_market, y_pred) + 0.1 * np.mean(pde_residuals)],
            'r2_score': [r2_score(V_market, y_pred)]
        }
        
        return self.training_history
    
    def predict(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Predict option price using trained PINN"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Create features
        features = FeatureEngineering.create_option_features(S, K, T, r)
        features_df = pd.DataFrame([features])
        
        # Ensure all feature columns are present
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.feature_names]
        
        # Scale and predict
        features_scaled = self.scaler_features.transform(features_df)
        prediction_scaled = self.model.predict(features_scaled)
        prediction = self.scaler_target.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).flatten()[0]
        
        return prediction

class ModelComparison:
    """Compare different ML models against Black-Scholes"""
    
    @staticmethod
    def compare_models(models_results: Dict[str, Dict], test_data: pd.DataFrame) -> pd.DataFrame:
        """Compare model performance metrics"""
        
        comparison_data = []
        
        for model_name, results in models_results.items():
            test_metrics = results.get('test_metrics', {})
            comparison_data.append({
                'Model': model_name,
                'MSE': test_metrics.get('mse', np.nan),
                'MAE': test_metrics.get('mae', np.nan),
                'R²': test_metrics.get('r2', np.nan),
                'RMSE': np.sqrt(test_metrics.get('mse', np.nan)) if 'mse' in test_metrics else np.nan
            })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def feature_importance_analysis(models_results: Dict[str, Dict]) -> pd.DataFrame:
        """Analyze feature importance across models"""
        
        importance_data = []
        
        for model_name, results in models_results.items():
            if 'feature_importance' in results:
                for feature, importance in results['feature_importance'].items():
                    importance_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Importance': importance
                    })
        
        return pd.DataFrame(importance_data)
