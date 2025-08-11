import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.black_scholes import BlackScholesModel, HestonModel
from models.volatility import GARCHModel, ImpliedVolatilitySurface, VolatilityModels
from models.monte_carlo import MonteCarloEngine, PathDependentOptions, VaRCalculator
from models.ml_models import MLOptionPricer, PhysicsInformedNeuralNetwork, FeatureEngineering
from models.exotic_options import ExoticOptionsEngine, StructuredProducts
from models.crypto_derivatives import CryptoDerivativesEngine, NFTDerivatives
from models.ai_enhanced_models import QuantumInspiredOptimizer, ReinforcementLearningTrader, TransformerPricePredictor, AutoMLFinancialModels
from models.real_time_risk_engine import RealTimeRiskEngine
from data.market_data import MarketDataProvider, RealTimeDataProvider, SyntheticDataGenerator, MarketIndicators
from data.alternative_data import AlternativeDataAggregator, SatelliteDataProvider, ESGDataProvider
from data.sentiment_analysis import SentimentAnalyzer, NewsDataProvider
from visualization.charts import FinancialCharts, InteractiveCharts
from visualization.volatility_surface import VolatilitySurfaceVisualizer
from backtesting.strategy import OptionTradingStrategy, PortfolioBacktester
from utils.calculations import FinancialCalculations, RiskMetrics
from config.settings import RISK_FREE_RATE, MONTE_CARLO_SIMULATIONS

# Configure Streamlit page
st.set_page_config(
    page_title="Quantitative Finance Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional financial styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin: 1rem 0;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .profit { color: #28a745; }
    .loss { color: #dc3545; }
    .warning-text { color: #ffc107; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Advanced Quantitative Finance Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    selected_page = st.sidebar.selectbox(
        "Select Analysis Module",
        [
            "🏠 Dashboard Overview",
            "📈 Option Pricing Models", 
            "🌊 Volatility Analysis",
            "🎲 Monte Carlo Simulations",
            "🤖 Machine Learning Models",
            "🧪 Exotic Options Lab",
            "₿ Crypto Derivatives",
            "🔮 AI-Enhanced Models",
            "🛰️ Alternative Data Analysis",
            "📰 Sentiment Analysis",
            "📊 Portfolio Backtesting",
            "🎯 Real-Time Risk Engine",
            "⚡ Quantum Portfolio Optimizer",
            "📋 Model Comparison"
        ]
    )
    
    # Route to selected page
    if selected_page == "🏠 Dashboard Overview":
        dashboard_overview()
    elif selected_page == "📈 Option Pricing Models":
        option_pricing_page()
    elif selected_page == "🌊 Volatility Analysis":
        volatility_analysis_page()
    elif selected_page == "🎲 Monte Carlo Simulations":
        monte_carlo_page()
    elif selected_page == "🤖 Machine Learning Models":
        ml_models_page()
    elif selected_page == "🧪 Exotic Options Lab":
        exotic_options_page()
    elif selected_page == "₿ Crypto Derivatives":
        crypto_derivatives_page()
    elif selected_page == "🔮 AI-Enhanced Models":
        ai_enhanced_models_page()
    elif selected_page == "🛰️ Alternative Data Analysis":
        alternative_data_page()
    elif selected_page == "📰 Sentiment Analysis":
        sentiment_analysis_page()
    elif selected_page == "📊 Portfolio Backtesting":
        backtesting_page()
    elif selected_page == "🎯 Real-Time Risk Engine":
        real_time_risk_page()
    elif selected_page == "⚡ Quantum Portfolio Optimizer":
        quantum_optimizer_page()
    elif selected_page == "📋 Model Comparison":
        model_comparison_page()

def dashboard_overview():
    """Dashboard overview with key metrics and market summary"""
    st.markdown('<h2 class="section-header">📊 Market Dashboard</h2>', unsafe_allow_html=True)
    
    # Market data provider
    market_provider = MarketDataProvider()
    real_time_provider = RealTimeDataProvider()
    
    # Sidebar inputs
    st.sidebar.markdown("### 🎯 Dashboard Settings")
    symbols = st.sidebar.multiselect(
        "Select Stocks",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "SPY"],
        default=["AAPL", "SPY"]
    )
    
    if not symbols:
        st.warning("Please select at least one symbol to display data.")
        return
    
    # Real-time quotes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Real-Time Market Data")
        quotes_data = []
        for symbol in symbols:
            quote = real_time_provider.get_real_time_quote(symbol)
            if quote:
                quotes_data.append(quote)
        
        if quotes_data:
            quotes_df = pd.DataFrame(quotes_data)
            
            # Display quotes with color coding
            for _, quote in quotes_df.iterrows():
                change_color = "profit" if quote['change'] >= 0 else "loss"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{quote['symbol']}</strong>: ${quote['price']:.2f} 
                    <span class="{change_color}">({quote['change']:+.2f}, {quote['change_percent']:+.2f}%)</span>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🌡️ Market Indicators")
        
        # VIX data
        vix_data = market_provider.get_vix_data()
        if not vix_data.empty:
            current_vix = vix_data['Close'].iloc[-1]
            vix_change = current_vix - vix_data['Close'].iloc[-2]
            vix_color = "loss" if current_vix > 20 else "profit"
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>VIX (Fear Index)</strong><br>
                <span class="{vix_color}">{current_vix:.2f} ({vix_change:+.2f})</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk-free rate
        risk_free = market_provider.get_risk_free_rate()
        st.markdown(f"""
        <div class="metric-card">
            <strong>10Y Treasury</strong><br>
            {risk_free*100:.2f}%
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    if symbols:
        st.subheader("📊 Price Charts")
        
        chart_tabs = st.tabs([f"{symbol}" for symbol in symbols])
        
        for i, symbol in enumerate(symbols):
            with chart_tabs[i]:
                data = market_provider.get_stock_data(symbol, period="6mo")
                if not data.empty:
                    # Create interactive chart
                    charts = FinancialCharts()
                    fig = charts.create_candlestick_chart(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        volatility = data['Volatility'].iloc[-1] if 'Volatility' in data.columns else 0
                        st.metric("Volatility (Ann.)", f"{volatility:.1%}")
                    with col2:
                        returns = data['Returns'].dropna()
                        sharpe = RiskMetrics.sharpe_ratio(returns) if len(returns) > 0 else 0
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    with col3:
                        max_dd = RiskMetrics.maximum_drawdown(data['Close'])
                        st.metric("Max Drawdown", f"{max_dd:.1%}")
                    with col4:
                        var_95 = RiskMetrics.value_at_risk(returns) if len(returns) > 0 else 0
                        st.metric("VaR (95%)", f"{var_95:.1%}")

def option_pricing_page():
    """Option pricing models page"""
    st.markdown('<h2 class="section-header">📈 Option Pricing Models</h2>', unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.markdown("### ⚙️ Option Parameters")
    
    # Symbol selection
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
    
    # Get current stock price
    market_provider = MarketDataProvider()
    stock_data = market_provider.get_stock_data(symbol, period="1mo")
    
    if stock_data.empty:
        st.error(f"Could not fetch data for {symbol}")
        return
    
    current_price = stock_data['Close'].iloc[-1]
    st.sidebar.info(f"Current {symbol} Price: ${current_price:.2f}")
    
    # Option parameters
    S = st.sidebar.number_input("Spot Price ($)", value=float(current_price), min_value=0.01)
    K = st.sidebar.number_input("Strike Price ($)", value=float(current_price), min_value=0.01)
    T = st.sidebar.slider("Time to Expiry (Years)", 0.01, 2.0, 0.25, 0.01)
    r = st.sidebar.slider("Risk-Free Rate", 0.0, 0.10, RISK_FREE_RATE, 0.001)
    sigma = st.sidebar.slider("Volatility", 0.01, 1.0, 0.25, 0.01)
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    
    # Model selection
    models = st.sidebar.multiselect(
        "Select Models", 
        ["Black-Scholes", "Heston", "Monte Carlo"],
        default=["Black-Scholes"]
    )
    
    # Calculate option prices
    results = {}
    
    if "Black-Scholes" in models:
        bs_price = BlackScholesModel.option_price(S, K, T, r, sigma, option_type)
        bs_greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, option_type)
        results["Black-Scholes"] = {"price": bs_price, "greeks": bs_greeks}
    
    if "Heston" in models:
        # Heston model parameters
        st.sidebar.markdown("#### Heston Parameters")
        v0 = st.sidebar.slider("Initial Variance", 0.01, 0.1, 0.04, 0.001)
        kappa = st.sidebar.slider("Mean Reversion Speed", 0.1, 5.0, 2.0, 0.1)
        theta = st.sidebar.slider("Long-term Variance", 0.01, 0.1, 0.04, 0.001)
        sigma_v = st.sidebar.slider("Vol of Vol", 0.1, 1.0, 0.3, 0.01)
        rho = st.sidebar.slider("Correlation", -1.0, 1.0, -0.7, 0.01)
        
        heston_price = HestonModel.option_price_fft(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type)
        results["Heston"] = {"price": heston_price, "greeks": {}}
    
    if "Monte Carlo" in models:
        mc_engine = MonteCarloEngine(n_simulations=10000)
        S_paths = mc_engine.geometric_brownian_motion(S, r, sigma, T)
        mc_result = mc_engine.price_european_option(S_paths, K, r, T, option_type)
        mc_greeks = mc_engine.calculate_greeks_fd(S, K, T, r, sigma, option_type)
        results["Monte Carlo"] = {"price": mc_result["price"], "greeks": mc_greeks}
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💰 Option Prices")
        for model, result in results.items():
            st.markdown(f"""
            <div class="metric-card">
                <strong>{model}</strong><br>
                ${result['price']:.4f}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔧 Option Greeks")
        if results:
            greeks_df = pd.DataFrame({
                model: result['greeks'] for model, result in results.items() 
                if result['greeks']
            }).T
            
            if not greeks_df.empty:
                st.dataframe(greeks_df.round(4))
    
    # Option chain
    st.subheader("⛓️ Option Chain Analysis")
    
    # Generate strikes around current price
    strikes = np.arange(S*0.8, S*1.2, S*0.05)
    option_chain = BlackScholesModel.create_option_chain(S, strikes.tolist(), T, r, sigma)
    
    # Display option chain
    st.dataframe(option_chain.round(4))
    
    # Sensitivity analysis
    st.subheader("📊 Sensitivity Analysis")
    sensitivity = BlackScholesModel.sensitivity_analysis(S, K, T, r, sigma, option_type)
    
    # Create sensitivity charts
    charts = FinancialCharts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_spot = charts.create_line_chart(
            sensitivity['spot_sensitivity'], 
            'Spot_Price', 'Option_Price',
            f"{option_type.title()} Option Price vs Spot Price"
        )
        st.plotly_chart(fig_spot, use_container_width=True)
    
    with col2:
        fig_vol = charts.create_line_chart(
            sensitivity['volatility_sensitivity'],
            'Volatility', 'Option_Price', 
            f"{option_type.title()} Option Price vs Volatility"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

def volatility_analysis_page():
    """Volatility analysis and surface construction"""
    st.markdown('<h2 class="section-header">🌊 Volatility Analysis</h2>', unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.markdown("### ⚙️ Volatility Settings")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
    
    # Get market data
    market_provider = MarketDataProvider()
    data = market_provider.get_stock_data(symbol, period="2y")
    
    if data.empty:
        st.error(f"Could not fetch data for {symbol}")
        return
    
    current_price = data['Close'].iloc[-1]
    returns = data['Returns'].dropna()
    
    # Volatility models
    vol_models = VolatilityModels()
    
    # Calculate different volatility measures
    realized_vol = vol_models.realized_volatility(returns, window=30)
    ewma_vol = vol_models.ewma_volatility(returns, lambda_param=0.94)
    
    # GARCH model
    garch_model = GARCHModel(returns)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Historical Volatility")
        
        # Create volatility chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=realized_vol.index,
            y=realized_vol.values,
            name="Realized Volatility (30d)",
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=ewma_vol.index,
            y=ewma_vol.values,
            name="EWMA Volatility",
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f"{symbol} Volatility Analysis",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Volatility Statistics")
        
        current_realized = realized_vol.iloc[-1] if len(realized_vol) > 0 else 0
        current_ewma = ewma_vol.iloc[-1] if len(ewma_vol) > 0 else 0
        
        st.metric("Current Realized Vol", f"{current_realized:.1%}")
        st.metric("Current EWMA Vol", f"{current_ewma:.1%}")
        st.metric("Vol of Vol", f"{returns.std() * np.sqrt(252):.1%}")
        
        # VIX comparison if SPY
        if symbol == "SPY":
            vix_data = market_provider.get_vix_data()
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1] / 100
                st.metric("VIX (Implied)", f"{current_vix:.1%}")
    
    # GARCH Analysis
    st.subheader("📈 GARCH Volatility Forecasting")
    
    if st.button("Fit GARCH Model"):
        with st.spinner("Fitting GARCH model..."):
            garch_results = garch_model.fit_garch(p=1, q=1)
            
            if garch_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**GARCH Model Parameters:**")
                    for param, value in garch_results['params'].items():
                        st.write(f"- {param}: {value:.6f}")
                    
                    st.write(f"**AIC:** {garch_results['aic']:.2f}")
                    st.write(f"**BIC:** {garch_results['bic']:.2f}")
                
                with col2:
                    # Forecast volatility
                    forecast = garch_model.forecast_volatility(horizon=30)
                    
                    if forecast:
                        forecast_df = pd.DataFrame({
                            'Date': pd.date_range(start=data.index[-1], periods=30, freq='D'),
                            'Volatility_Forecast': forecast['volatility_forecast']
                        })
                        
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Volatility_Forecast'],
                            name="GARCH Forecast",
                            line=dict(color='green')
                        ))
                        
                        fig_forecast.update_layout(
                            title="30-Day Volatility Forecast",
                            xaxis_title="Date",
                            yaxis_title="Annualized Volatility"
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Implied Volatility Surface
    st.subheader("🏔️ Implied Volatility Surface")
    
    # Create synthetic IV surface
    iv_surface = ImpliedVolatilitySurface()
    
    strikes = np.arange(current_price * 0.8, current_price * 1.2, current_price * 0.05)
    expiries = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # 3M, 6M, 9M, 1Y, 1.5Y, 2Y
    
    surface_data = iv_surface.create_synthetic_surface(
        current_price, strikes.tolist(), expiries, base_vol=current_realized
    )
    
    # Create 3D surface plot
    surface_viz = VolatilitySurfaceVisualizer()
    fig_3d = surface_viz.create_3d_surface(surface_data)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Surface analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility smile for selected expiry
        selected_expiry = st.selectbox("Select Expiry for Smile", expiries)
        smile_data = surface_data[surface_data['Expiry'] == selected_expiry]
        
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=smile_data['Moneyness'],
            y=smile_data['ImpliedVol'],
            mode='lines+markers',
            name=f"Volatility Smile ({selected_expiry}Y)"
        ))
        
        fig_smile.update_layout(
            title="Volatility Smile",
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Implied Volatility"
        )
        
        st.plotly_chart(fig_smile, use_container_width=True)
    
    with col2:
        # Term structure for ATM options
        atm_data = surface_data[abs(surface_data['Moneyness'] - 1.0) < 0.05]
        
        fig_term = go.Figure()
        fig_term.add_trace(go.Scatter(
            x=atm_data['Expiry'],
            y=atm_data['ImpliedVol'],
            mode='lines+markers',
            name="ATM Volatility Term Structure"
        ))
        
        fig_term.update_layout(
            title="Volatility Term Structure",
            xaxis_title="Time to Expiry (Years)",
            yaxis_title="Implied Volatility"
        )
        
        st.plotly_chart(fig_term, use_container_width=True)

def monte_carlo_page():
    """Monte Carlo simulations page"""
    st.markdown('<h2 class="section-header">🎲 Monte Carlo Simulations</h2>', unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.markdown("### ⚙️ Simulation Parameters")
    
    # Basic parameters
    S0 = st.sidebar.number_input("Initial Stock Price", value=100.0, min_value=0.01)
    mu = st.sidebar.slider("Drift (μ)", -0.2, 0.3, 0.1, 0.01)
    sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.25, 0.01)
    T = st.sidebar.slider("Time Horizon (Years)", 0.1, 2.0, 1.0, 0.1)
    n_simulations = st.sidebar.selectbox("Number of Simulations", [1000, 5000, 10000, 50000], index=2)
    n_steps = st.sidebar.selectbox("Time Steps", [50, 100, 252, 500], index=2)
    
    # Model selection
    simulation_type = st.sidebar.selectbox(
        "Simulation Model",
        ["Geometric Brownian Motion", "Heston Stochastic Volatility", "Jump Diffusion"]
    )
    
    # Initialize Monte Carlo engine
    mc_engine = MonteCarloEngine(n_simulations=n_simulations, n_steps=n_steps)
    
    # Generate simulations based on selected model
    if simulation_type == "Geometric Brownian Motion":
        if st.button("Run GBM Simulation"):
            with st.spinner("Running Monte Carlo simulation..."):
                S_paths = mc_engine.geometric_brownian_motion(S0, mu, sigma, T)
                
                # Display results
                display_simulation_results(S_paths, S0, mu, sigma, T, "GBM")
    
    elif simulation_type == "Heston Stochastic Volatility":
        # Heston parameters
        st.sidebar.markdown("#### Heston Parameters")
        v0 = st.sidebar.slider("Initial Variance", 0.01, 0.1, 0.04, 0.001)
        kappa = st.sidebar.slider("Mean Reversion Speed", 0.1, 5.0, 2.0, 0.1)
        theta = st.sidebar.slider("Long-term Variance", 0.01, 0.1, 0.04, 0.001)
        sigma_v = st.sidebar.slider("Vol of Vol", 0.1, 1.0, 0.3, 0.01)
        rho = st.sidebar.slider("Correlation", -1.0, 1.0, -0.7, 0.01)
        
        if st.button("Run Heston Simulation"):
            with st.spinner("Running Heston simulation..."):
                S_paths, v_paths = mc_engine.heston_simulation(S0, v0, mu, kappa, theta, sigma_v, rho, T)
                
                # Display results
                display_simulation_results(S_paths, S0, mu, sigma, T, "Heston", v_paths)
    
    elif simulation_type == "Jump Diffusion":
        # Jump parameters
        st.sidebar.markdown("#### Jump Parameters")
        lambda_j = st.sidebar.slider("Jump Intensity", 0.0, 2.0, 0.5, 0.1)
        mu_j = st.sidebar.slider("Jump Mean", -0.2, 0.2, -0.05, 0.01)
        sigma_j = st.sidebar.slider("Jump Volatility", 0.01, 0.5, 0.1, 0.01)
        
        if st.button("Run Jump Diffusion Simulation"):
            with st.spinner("Running jump diffusion simulation..."):
                S_paths = mc_engine.jump_diffusion_simulation(S0, mu, sigma, lambda_j, mu_j, sigma_j, T)
                
                # Display results
                display_simulation_results(S_paths, S0, mu, sigma, T, "Jump Diffusion")
    
    # Option pricing section
    st.subheader("💰 Option Pricing with Monte Carlo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### European Options")
        K_european = st.number_input("Strike Price", value=S0, min_value=0.01, key="european_strike")
        option_type = st.selectbox("Option Type", ["call", "put"])
        r = st.slider("Risk-Free Rate", 0.0, 0.1, 0.05, 0.001)
        
        if st.button("Price European Option"):
            S_paths = mc_engine.geometric_brownian_motion(S0, r, sigma, T)
            option_result = mc_engine.price_european_option(S_paths, K_european, r, T, option_type)
            
            st.success(f"Option Price: ${option_result['price']:.4f}")
            st.info(f"95% Confidence Interval: [${option_result['confidence_interval'][0]:.4f}, ${option_result['confidence_interval'][1]:.4f}]")
            
            # Payoff distribution
            fig_payoff = go.Figure()
            fig_payoff.add_histogram(x=option_result['payoffs'], bins=50, name="Payoff Distribution")
            fig_payoff.update_layout(
                title=f"{option_type.title()} Option Payoff Distribution",
                xaxis_title="Payoff",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_payoff, use_container_width=True)
    
    with col2:
        st.markdown("#### American Options (LSM)")
        K_american = st.number_input("Strike Price", value=S0, min_value=0.01, key="american_strike")
        american_type = st.selectbox("American Option Type", ["put", "call"])
        
        if st.button("Price American Option"):
            S_paths = mc_engine.geometric_brownian_motion(S0, r, sigma, T)
            american_result = mc_engine.price_american_option_lsm(S_paths, K_american, r, T, american_type)
            
            st.success(f"American Option Price: ${american_result['price']:.4f}")
            st.info(f"Standard Error: ${american_result['std_error']:.4f}")
    
    # Path-dependent options
    st.subheader("🛤️ Path-Dependent Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Asian Options")
        K_asian = st.number_input("Asian Strike", value=S0, min_value=0.01)
        asian_type = st.selectbox("Asian Type", ["call", "put"])
        average_type = st.selectbox("Average Type", ["arithmetic", "geometric"])
        
        if st.button("Price Asian Option"):
            S_paths = mc_engine.geometric_brownian_motion(S0, r, sigma, T)
            asian_result = PathDependentOptions.asian_option(S_paths, K_asian, r, T, asian_type, average_type)
            
            st.success(f"Asian Option Price: ${asian_result['price']:.4f}")
    
    with col2:
        st.markdown("#### Barrier Options")
        K_barrier = st.number_input("Barrier Strike", value=S0, min_value=0.01)
        barrier_level = st.number_input("Barrier Level", value=S0*1.2, min_value=0.01)
        barrier_type = st.selectbox("Barrier Type", ["up_and_out", "down_and_out", "up_and_in", "down_and_in"])
        barrier_option_type = st.selectbox("Barrier Option Type", ["call", "put"])
        
        if st.button("Price Barrier Option"):
            S_paths = mc_engine.geometric_brownian_motion(S0, r, sigma, T)
            barrier_result = PathDependentOptions.barrier_option(
                S_paths, K_barrier, barrier_level, r, T, barrier_option_type, barrier_type
            )
            
            st.success(f"Barrier Option Price: ${barrier_result['price']:.4f}")
            st.info(f"Barrier Hit Ratio: {barrier_result['barrier_hit_ratio']:.1%}")

def display_simulation_results(S_paths, S0, mu, sigma, T, model_name, v_paths=None):
    """Display Monte Carlo simulation results"""
    
    # Summary statistics
    final_prices = S_paths[:, -1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
    with col2:
        st.metric("Std Dev", f"${np.std(final_prices):.2f}")
    with col3:
        st.metric("95% VaR", f"${np.percentile(final_prices, 5):.2f}")
    with col4:
        returns = (final_prices - S0) / S0
        positive_returns = np.mean(returns > 0)
        st.metric("Prob(Positive)", f"{positive_returns:.1%}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price paths (sample)
        fig_paths = go.Figure()
        
        # Show first 100 paths for visualization
        n_display = min(100, S_paths.shape[0])
        time_grid = np.linspace(0, T, S_paths.shape[1])
        
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(
                x=time_grid,
                y=S_paths[i, :],
                mode='lines',
                line=dict(width=0.5, color='rgba(0,100,200,0.1)'),
                showlegend=False
            ))
        
        # Add mean path
        mean_path = np.mean(S_paths, axis=0)
        fig_paths.add_trace(go.Scatter(
            x=time_grid,
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Mean Path'
        ))
        
        fig_paths.update_layout(
            title=f"{model_name} Price Paths",
            xaxis_title="Time (Years)",
            yaxis_title="Stock Price"
        )
        
        st.plotly_chart(fig_paths, use_container_width=True)
    
    with col2:
        # Final price distribution
        fig_dist = go.Figure()
        fig_dist.add_histogram(
            x=final_prices, 
            bins=50, 
            name="Final Price Distribution",
            histnorm='probability density'
        )
        
        # Add theoretical normal distribution for comparison
        if model_name == "GBM":
            theoretical_mean = S0 * np.exp(mu * T)
            theoretical_std = S0 * np.sqrt(np.exp(2*mu*T) * (np.exp(sigma**2 * T) - 1))
            
            x_theory = np.linspace(final_prices.min(), final_prices.max(), 100)
            y_theory = (1/(theoretical_std * np.sqrt(2*np.pi))) * \
                      np.exp(-0.5 * ((x_theory - theoretical_mean)/theoretical_std)**2)
            
            fig_dist.add_trace(go.Scatter(
                x=x_theory,
                y=y_theory,
                mode='lines',
                name='Theoretical Normal',
                line=dict(color='red', dash='dash')
            ))
        
        fig_dist.update_layout(
            title="Final Price Distribution",
            xaxis_title="Final Stock Price",
            yaxis_title="Density"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Volatility paths for Heston
    if v_paths is not None:
        st.subheader("📊 Volatility Analysis")
        
        # Volatility paths
        fig_vol = go.Figure()
        
        n_display = min(50, v_paths.shape[0])
        time_grid = np.linspace(0, T, v_paths.shape[1])
        
        for i in range(n_display):
            fig_vol.add_trace(go.Scatter(
                x=time_grid,
                y=np.sqrt(v_paths[i, :]),
                mode='lines',
                line=dict(width=0.5, color='rgba(200,100,0,0.1)'),
                showlegend=False
            ))
        
        # Add mean volatility path
        mean_vol = np.sqrt(np.mean(v_paths, axis=0))
        fig_vol.add_trace(go.Scatter(
            x=time_grid,
            y=mean_vol,
            mode='lines',
            line=dict(width=3, color='orange'),
            name='Mean Volatility'
        ))
        
        fig_vol.update_layout(
            title="Stochastic Volatility Paths",
            xaxis_title="Time (Years)",
            yaxis_title="Volatility"
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)

def ml_models_page():
    """Machine learning models page"""
    st.markdown('<h2 class="section-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### ⚙️ ML Model Settings")
    
    # Model selection
    ml_model_type = st.sidebar.selectbox(
        "Select ML Model",
        ["XGBoost", "Neural Network", "Random Forest", "Physics-Informed NN"]
    )
    
    # Generate synthetic training data
    st.subheader("📊 Training Data Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_observations = st.number_input("Number of Training Samples", value=5000, min_value=100, max_value=20000)
        base_price = st.number_input("Base Stock Price", value=100.0, min_value=1.0)
        base_vol = st.slider("Base Volatility", 0.1, 0.5, 0.25, 0.01)
        risk_free = st.slider("Risk-Free Rate", 0.0, 0.1, 0.05, 0.001)
    
    with col2:
        # Strike range
        st.write("**Strike Price Range:**")
        strike_min = st.slider("Min Strike (% of Spot)", 0.5, 1.0, 0.8, 0.05)
        strike_max = st.slider("Max Strike (% of Spot)", 1.0, 1.5, 1.2, 0.05)
        
        # Expiry range
        st.write("**Time to Expiry Range:**")
        expiry_min = st.slider("Min Expiry (Years)", 0.01, 0.5, 0.1, 0.01)
        expiry_max = st.slider("Max Expiry (Years)", 0.5, 2.0, 1.0, 0.1)
    
    if st.button("Generate Training Data"):
        with st.spinner("Generating synthetic option market data..."):
            # Generate strikes and expiries
            strikes = np.linspace(base_price * strike_min, base_price * strike_max, 10)
            expiries = np.linspace(expiry_min, expiry_max, 8)
            
            # Generate synthetic market data
            synthetic_generator = SyntheticDataGenerator()
            market_data = synthetic_generator.generate_option_market_data(
                base_price, strikes.tolist(), expiries.tolist(), risk_free, base_vol, n_observations
            )
            
            st.session_state['market_data'] = market_data
            st.success(f"Generated {len(market_data)} training samples")
            
            # Show data preview
            st.write("**Data Preview:**")
            st.dataframe(market_data.head(10))
            
            # Data statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Call Options", len(market_data[market_data['option_type'] == 'call']))
            with col2:
                st.metric("Put Options", len(market_data[market_data['option_type'] == 'put']))
            with col3:
                st.metric("Unique Strikes", market_data['strike_price'].nunique())
    
    # Model training section
    if 'market_data' in st.session_state:
        st.subheader("🎯 Model Training")
        
        market_data = st.session_state['market_data']
        ml_pricer = MLOptionPricer()
        
        # Prepare training data
        X, y = ml_pricer.prepare_training_data(market_data, 'option_price')
        
        if ml_model_type == "XGBoost":
            if st.button("Train XGBoost Model"):
                with st.spinner("Training XGBoost model..."):
                    xgb_results = ml_pricer.train_xgboost_model(X, y, hyperparameter_tuning=False)
                    
                    st.session_state['xgb_results'] = xgb_results
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Training Metrics:**")
                        st.json(xgb_results['train_metrics'])
                    
                    with col2:
                        st.write("**Test Metrics:**")
                        st.json(xgb_results['test_metrics'])
                    
                    # Feature importance
                    feature_importance = pd.DataFrame(
                        list(xgb_results['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='Importance', y='Feature',
                        orientation='h',
                        title="Top 10 Feature Importance"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        elif ml_model_type == "Neural Network":
            # NN architecture settings
            st.write("**Neural Network Architecture:**")
            layers = []
            n_layers = st.slider("Number of Hidden Layers", 1, 5, 3)
            
            for i in range(n_layers):
                layer_size = st.slider(f"Layer {i+1} Size", 10, 200, 100-i*20, 10)
                layers.append(layer_size)
            
            if st.button("Train Neural Network"):
                with st.spinner("Training neural network..."):
                    nn_results = ml_pricer.train_neural_network(X, y, architecture=layers)
                    
                    st.session_state['nn_results'] = nn_results
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Training Metrics:**")
                        st.json(nn_results['train_metrics'])
                    
                    with col2:
                        st.write("**Test Metrics:**")
                        st.json(nn_results['test_metrics'])
                    
                    # Training history
                    history = nn_results['history']
                    
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(
                        y=history['loss'], name='Training Loss', mode='lines'
                    ))
                    fig_history.add_trace(go.Scatter(
                        y=history['val_loss'], name='Validation Loss', mode='lines'
                    ))
                    
                    fig_history.update_layout(
                        title="Training History",
                        xaxis_title="Epoch",
                        yaxis_title="Loss"
                    )
                    
                    st.plotly_chart(fig_history, use_container_width=True)
        
        elif ml_model_type == "Random Forest":
            if st.button("Train Random Forest"):
                with st.spinner("Training Random Forest model..."):
                    rf_results = ml_pricer.train_random_forest(X, y)
                    
                    st.session_state['rf_results'] = rf_results
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Training Metrics:**")
                        st.json(rf_results['train_metrics'])
                    
                    with col2:
                        st.write("**Test Metrics:**")
                        st.json(rf_results['test_metrics'])
        
        elif ml_model_type == "Physics-Informed NN":
            st.write("**Physics-Informed Neural Network (PINN)**")
            st.info("This model incorporates the Black-Scholes PDE as a physics constraint during training.")
            
            # PINN architecture
            pinn_layers = st.multiselect(
                "Hidden Layer Sizes",
                [10, 20, 30, 50, 100],
                default=[20, 20, 20]
            )
            
            epochs = st.slider("Training Epochs", 100, 2000, 1000, 100)
            
            if st.button("Train PINN Model"):
                with st.spinner("Training Physics-Informed Neural Network..."):
                    # Prepare data for PINN
                    pinn = PhysicsInformedNeuralNetwork(layers=pinn_layers + [1])
                    
                    # Extract features for PINN training
                    S_data = market_data['spot_price'].values
                    K_data = market_data['strike_price'].values
                    T_data = market_data['time_to_expiry'].values
                    r_data = market_data['risk_free_rate'].values
                    sigma_data = market_data['implied_volatility'].values
                    V_market = market_data['option_price'].values
                    
                    # Train PINN
                    history = pinn.train_pinn(
                        S_data, K_data, T_data, r_data, sigma_data, V_market, epochs
                    )
                    
                    st.session_state['pinn_model'] = pinn
                    st.session_state['pinn_history'] = history
                    
                    # Display training progress
                    fig_pinn = go.Figure()
                    fig_pinn.add_trace(go.Scatter(
                        y=history['total_loss'], name='Total Loss', mode='lines'
                    ))
                    fig_pinn.add_trace(go.Scatter(
                        y=history['data_loss'], name='Data Loss', mode='lines'
                    ))
                    fig_pinn.add_trace(go.Scatter(
                        y=history['pde_loss'], name='PDE Loss', mode='lines'
                    ))
                    
                    fig_pinn.update_layout(
                        title="PINN Training History",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        yaxis_type="log"
                    )
                    
                    st.plotly_chart(fig_pinn, use_container_width=True)
        
        # Model prediction section
        st.subheader("🎯 Model Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input parameters for prediction
            pred_S = st.number_input("Stock Price", value=100.0, min_value=0.01)
            pred_K = st.number_input("Strike Price", value=100.0, min_value=0.01)
            pred_T = st.slider("Time to Expiry", 0.01, 2.0, 0.25, 0.01)
            pred_r = st.slider("Risk-Free Rate", 0.0, 0.1, 0.05, 0.001)
        
        with col2:
            # Make predictions with trained models
            if st.button("Make Predictions"):
                predictions = {}
                
                # Black-Scholes benchmark
                bs_call = BlackScholesModel.option_price(pred_S, pred_K, pred_T, pred_r, base_vol, 'call')
                bs_put = BlackScholesModel.option_price(pred_S, pred_K, pred_T, pred_r, base_vol, 'put')
                predictions['Black-Scholes Call'] = bs_call
                predictions['Black-Scholes Put'] = bs_put
                
                # ML model predictions
                if 'xgb_results' in st.session_state:
                    try:
                        xgb_call = ml_pricer.predict('xgboost', pred_S, pred_K, pred_T, pred_r)
                        predictions['XGBoost Call'] = xgb_call
                    except:
                        st.warning("XGBoost prediction failed")
                
                if 'pinn_model' in st.session_state:
                    try:
                        pinn = st.session_state['pinn_model']
                        pinn_call = pinn.predict(pred_S, pred_K, pred_T, pred_r, base_vol)
                        predictions['PINN Call'] = pinn_call
                    except:
                        st.warning("PINN prediction failed")
                
                # Display predictions
                for model, price in predictions.items():
                    st.write(f"**{model}:** ${price:.4f}")
    
    else:
        st.info("Please generate training data first to proceed with model training.")

def sentiment_analysis_page():
    """Sentiment analysis page"""
    st.markdown('<h2 class="section-header">📰 Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    # Initialize providers
    sentiment_analyzer = SentimentAnalyzer()
    news_provider = NewsDataProvider()
    
    st.sidebar.markdown("### ⚙️ Sentiment Settings")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Stock Sentiment", "Market Sentiment", "News Analysis", "Social Media Trends"]
    )
    
    if analysis_type == "Stock Sentiment":
        # Stock-specific sentiment analysis
        symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 {symbol} Sentiment Analysis")
            
            if st.button("Analyze Stock Sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    # Get news sentiment
                    news_sentiment = sentiment_analyzer.analyze_stock_sentiment(symbol)
                    
                    if news_sentiment:
                        # Display sentiment metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            sentiment_score = news_sentiment['overall_sentiment']
                            sentiment_color = "profit" if sentiment_score > 0 else "loss"
                            st.markdown(f"""
                            <div class="metric-card">
                                <strong>Overall Sentiment</strong><br>
                                <span class="{sentiment_color}">{sentiment_score:.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Positive Articles", f"{news_sentiment['positive_count']}")
                        
                        with col3:
                            st.metric("Negative Articles", f"{news_sentiment['negative_count']}")
                        
                        # Sentiment over time
                        if 'sentiment_history' in news_sentiment:
                            sentiment_df = pd.DataFrame(news_sentiment['sentiment_history'])
                            
                            fig_sentiment = go.Figure()
                            fig_sentiment.add_trace(go.Scatter(
                                x=sentiment_df['date'],
                                y=sentiment_df['sentiment_score'],
                                mode='lines+markers',
                                name='Sentiment Score',
                                line=dict(color='blue')
                            ))
                            
                            fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            fig_sentiment.update_layout(
                                title=f"{symbol} Sentiment Trend",
                                xaxis_title="Date",
                                yaxis_title="Sentiment Score",
                                yaxis=dict(range=[-1, 1])
                            )
                            
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        # Top articles
                        if 'articles' in news_sentiment:
                            st.subheader("📰 Recent Articles")
                            
                            for article in news_sentiment['articles'][:5]:
                                sentiment_emoji = "🟢" if article['sentiment'] > 0.1 else "🔴" if article['sentiment'] < -0.1 else "🟡"
                                st.markdown(f"""
                                **{sentiment_emoji} {article['title']}**  
                                *Sentiment: {article['sentiment']:.2f}*  
                                {article['summary'][:200]}...  
                                [Read more]({article['url']})
                                
                                ---
                                """)
        
        with col2:
            st.subheader("📊 Sentiment Metrics")
            
            # Social media sentiment simulation
            social_sentiment = sentiment_analyzer.get_social_media_sentiment(symbol)
            
            if social_sentiment:
                # Twitter sentiment
                twitter_score = social_sentiment.get('twitter_sentiment', 0)
                twitter_color = "profit" if twitter_score > 0 else "loss"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Twitter Sentiment</strong><br>
                    <span class="{twitter_color}">{twitter_score:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Reddit sentiment
                reddit_score = social_sentiment.get('reddit_sentiment', 0)
                reddit_color = "profit" if reddit_score > 0 else "loss"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Reddit Sentiment</strong><br>
                    <span class="{reddit_color}">{reddit_score:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Sentiment volume
                st.metric("Daily Mentions", social_sentiment.get('mention_count', 0))
                st.metric("Sentiment Volume", social_sentiment.get('sentiment_volume', 0))
    
    elif analysis_type == "Market Sentiment":
        st.subheader("🌍 Overall Market Sentiment")
        
        # Market-wide sentiment analysis
        if st.button("Analyze Market Sentiment"):
            with st.spinner("Analyzing market sentiment..."):
                market_sentiment = sentiment_analyzer.analyze_market_sentiment()
                
                if market_sentiment:
                    # Market sentiment gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = market_sentiment['overall_score'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Market Sentiment Score"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "lightgreen"},
                                {'range': [75, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Sector sentiment breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sector_sentiment = market_sentiment.get('sector_sentiment', {})
                        
                        if sector_sentiment:
                            sectors_df = pd.DataFrame(
                                list(sector_sentiment.items()),
                                columns=['Sector', 'Sentiment']
                            ).sort_values('Sentiment', ascending=True)
                            
                            fig_sectors = px.bar(
                                sectors_df,
                                x='Sentiment', y='Sector',
                                orientation='h',
                                title="Sector Sentiment Breakdown",
                                color='Sentiment',
                                color_continuous_scale='RdYlGn'
                            )
                            
                            st.plotly_chart(fig_sectors, use_container_width=True)
                    
                    with col2:
                        # Fear & Greed indicators
                        st.subheader("😨 Fear & Greed Indicators")
                        
                        indicators = market_sentiment.get('fear_greed_indicators', {})
                        
                        for indicator, value in indicators.items():
                            if value > 0.6:
                                emoji = "🟢"
                                status = "Greedy"
                            elif value < 0.4:
                                emoji = "🔴" 
                                status = "Fearful"
                            else:
                                emoji = "🟡"
                                status = "Neutral"
                            
                            st.markdown(f"**{emoji} {indicator}:** {status} ({value:.2f})")
    
    elif analysis_type == "News Analysis":
        st.subheader("📰 Financial News Analysis")
        
        # News source selection
        news_sources = st.multiselect(
            "Select News Sources",
            ["Bloomberg", "Reuters", "CNBC", "MarketWatch", "Financial Times"],
            default=["Bloomberg", "Reuters"]
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        if st.button("Analyze Financial News"):
            with st.spinner("Analyzing financial news..."):
                news_analysis = news_provider.analyze_financial_news(
                    sources=news_sources,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if news_analysis:
                    # News sentiment timeline
                    if 'daily_sentiment' in news_analysis:
                        sentiment_timeline = pd.DataFrame(news_analysis['daily_sentiment'])
                        
                        fig_timeline = go.Figure()
                        fig_timeline.add_trace(go.Scatter(
                            x=pd.to_datetime(sentiment_timeline['date']),
                            y=sentiment_timeline['sentiment'],
                            mode='lines+markers',
                            name='Daily Sentiment',
                            fill='tonexty'
                        ))
                        
                        fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        fig_timeline.update_layout(
                            title="Financial News Sentiment Timeline",
                            xaxis_title="Date",
                            yaxis_title="Average Sentiment Score"
                        )
                        
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Topic analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'topic_sentiment' in news_analysis:
                            topics_df = pd.DataFrame(
                                news_analysis['topic_sentiment'].items(),
                                columns=['Topic', 'Sentiment']
                            )
                            
                            fig_topics = px.bar(
                                topics_df,
                                x='Topic', y='Sentiment',
                                title="Sentiment by Topic",
                                color='Sentiment',
                                color_continuous_scale='RdYlGn'
                            )
                            
                            st.plotly_chart(fig_topics, use_container_width=True)
                    
                    with col2:
                        if 'key_themes' in news_analysis:
                            st.subheader("🏷️ Key Themes")
                            
                            for theme in news_analysis['key_themes'][:10]:
                                st.markdown(f"• {theme}")
    
    elif analysis_type == "Social Media Trends":
        st.subheader("📱 Social Media Sentiment Trends")
        
        # Platform selection
        platforms = st.multiselect(
            "Select Platforms",
            ["Twitter", "Reddit", "StockTwits", "Discord"],
            default=["Twitter", "Reddit"]
        )
        
        # Keywords
        keywords = st.text_input("Keywords (comma-separated)", value="stocks, market, trading, options")
        
        if st.button("Analyze Social Media Trends"):
            with st.spinner("Analyzing social media trends..."):
                social_trends = sentiment_analyzer.analyze_social_media_trends(
                    platforms=platforms,
                    keywords=keywords.split(',') if keywords else []
                )
                
                if social_trends:
                    # Platform comparison
                    platform_data = []
                    for platform in platforms:
                        platform_sentiment = social_trends.get(f'{platform.lower()}_sentiment', {})
                        platform_data.append({
                            'Platform': platform,
                            'Sentiment': platform_sentiment.get('average_sentiment', 0),
                            'Volume': platform_sentiment.get('post_count', 0),
                            'Engagement': platform_sentiment.get('engagement_score', 0)
                        })
                    
                    platform_df = pd.DataFrame(platform_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_sentiment = px.bar(
                            platform_df,
                            x='Platform', y='Sentiment',
                            title="Platform Sentiment Comparison",
                            color='Sentiment',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    with col2:
                        fig_volume = px.bar(
                            platform_df,
                            x='Platform', y='Volume',
                            title="Discussion Volume by Platform"
                        )
                        st.plotly_chart(fig_volume, use_container_width=True)
                    
                    # Trending topics
                    if 'trending_topics' in social_trends:
                        st.subheader("🔥 Trending Topics")
                        
                        trending_df = pd.DataFrame(social_trends['trending_topics'])
                        
                        fig_trending = px.treemap(
                            trending_df,
                            path=['topic'],
                            values='mention_count',
                            color='sentiment_score',
                            color_continuous_scale='RdYlGn',
                            title="Trending Topics by Sentiment"
                        )
                        
                        st.plotly_chart(fig_trending, use_container_width=True)

def alternative_data_page():
    """Alternative data analysis page"""
    st.markdown('<h2 class="section-header">🛰️ Alternative Data Analysis</h2>', unsafe_allow_html=True)
    
    # Initialize providers
    alt_data_aggregator = AlternativeDataAggregator()
    satellite_provider = SatelliteDataProvider()
    esg_provider = ESGDataProvider()
    
    st.sidebar.markdown("### ⚙️ Alternative Data Settings")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Satellite Imagery", "ESG Metrics", "Economic Indicators", "Company Analysis"]
    )
    
    if data_source == "Satellite Imagery":
        st.subheader("🛰️ Satellite Data Analysis")
        
        # Analysis type
        satellite_analysis = st.selectbox(
            "Analysis Type",
            ["Retail Footfall", "Construction Activity", "Agricultural Yields", "Oil Storage"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if satellite_analysis == "Retail Footfall":
                location = st.text_input("Location", value="Times Square, NYC")
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
                end_date = st.date_input("End Date", value=datetime.now())
                
                if st.button("Analyze Retail Footfall"):
                    with st.spinner("Analyzing satellite data..."):
                        footfall_data = satellite_provider.get_retail_footfall_data(
                            location, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                        )
                        
                        if not footfall_data.empty:
                            # Footfall trend chart
                            fig_footfall = go.Figure()
                            
                            fig_footfall.add_trace(go.Scatter(
                                x=footfall_data.index,
                                y=footfall_data['footfall_count'],
                                mode='lines',
                                name='Daily Footfall',
                                line=dict(color='blue')
                            ))
                            
                            fig_footfall.add_trace(go.Scatter(
                                x=footfall_data.index,
                                y=footfall_data['footfall_ma_7'],
                                mode='lines',
                                name='7-Day MA',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig_footfall.update_layout(
                                title=f"Retail Footfall Analysis - {location}",
                                xaxis_title="Date",
                                yaxis_title="Footfall Count"
                            )
                            
                            st.plotly_chart(fig_footfall, use_container_width=True)
                            
                            # Key metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_footfall = footfall_data['footfall_count'].mean()
                                st.metric("Average Daily Footfall", f"{avg_footfall:,.0f}")
                            with col2:
                                peak_footfall = footfall_data['footfall_count'].max()
                                st.metric("Peak Footfall", f"{peak_footfall:,.0f}")
                            with col3:
                                growth_rate = (footfall_data['footfall_count'].tail(7).mean() / 
                                             footfall_data['footfall_count'].head(7).mean() - 1) * 100
                                st.metric("Growth Rate", f"{growth_rate:+.1f}%")
            
            elif satellite_analysis == "Oil Storage":
                facility = st.text_input("Storage Facility", value="Cushing, OK")
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=60), key="oil_start")
                end_date = st.date_input("End Date", value=datetime.now(), key="oil_end")
                
                if st.button("Analyze Oil Storage"):
                    with st.spinner("Analyzing oil storage levels..."):
                        storage_data = satellite_provider.get_oil_storage_levels(
                            facility, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                        )
                        
                        if not storage_data.empty:
                            # Storage level chart
                            fig_storage = go.Figure()
                            
                            fig_storage.add_trace(go.Scatter(
                                x=storage_data.index,
                                y=storage_data['storage_level_pct'],
                                mode='lines+markers',
                                name='Storage Level %',
                                line=dict(color='orange')
                            ))
                            
                            fig_storage.add_hline(y=80, line_dash="dash", line_color="red", 
                                                annotation_text="High Capacity")
                            fig_storage.add_hline(y=20, line_dash="dash", line_color="blue",
                                                annotation_text="Low Capacity")
                            
                            fig_storage.update_layout(
                                title=f"Oil Storage Analysis - {facility}",
                                xaxis_title="Date",
                                yaxis_title="Storage Level (%)",
                                yaxis=dict(range=[0, 100])
                            )
                            
                            st.plotly_chart(fig_storage, use_container_width=True)
        
        with col2:
            if satellite_analysis == "Construction Activity":
                region = st.text_input("Region", value="Manhattan, NYC")
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=180), key="construction_start")
                end_date = st.date_input("End Date", value=datetime.now(), key="construction_end")
                
                if st.button("Analyze Construction Activity"):
                    with st.spinner("Analyzing construction activity..."):
                        construction_data = satellite_provider.get_construction_activity(
                            region, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                        )
                        
                        if not construction_data.empty:
                            # Construction activity chart
                            fig_construction = go.Figure()
                            
                            fig_construction.add_trace(go.Scatter(
                                x=construction_data.index,
                                y=construction_data['construction_activity_index'],
                                mode='lines+markers',
                                name='Activity Index',
                                line=dict(color='green')
                            ))
                            
                            fig_construction.update_layout(
                                title=f"Construction Activity - {region}",
                                xaxis_title="Date",
                                yaxis_title="Activity Index"
                            )
                            
                            st.plotly_chart(fig_construction, use_container_width=True)
                            
                            # Summary metrics
                            total_new_sites = construction_data['new_construction_sites'].sum()
                            total_completed = construction_data['completed_projects'].sum()
                            avg_cranes = construction_data['active_crane_count'].mean()
                            
                            st.metric("New Construction Sites", f"{total_new_sites}")
                            st.metric("Completed Projects", f"{total_completed}")
                            st.metric("Average Active Cranes", f"{avg_cranes:.0f}")
    
    elif data_source == "ESG Metrics":
        st.subheader("🌱 ESG Data Analysis")
        
        company = st.text_input("Company Name", value="Apple Inc.")
        year = st.selectbox("Year", [2024, 2023, 2022, 2021], index=0)
        
        if st.button("Analyze ESG Metrics"):
            with st.spinner("Analyzing ESG data..."):
                esg_data = esg_provider.get_carbon_emissions_data(company, year)
                supply_risk = esg_provider.get_supply_chain_risk_data(company)
                
                if esg_data:
                    # ESG Overview
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        esg_score = esg_data['esg_score']
                        score_color = "profit" if esg_score > 70 else "loss" if esg_score < 40 else "warning-text"
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>Overall ESG Score</strong><br>
                            <span class="{score_color}">{esg_score:.0f}/100</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Total Emissions", f"{esg_data['total_emissions']:,.0f} tons CO2e")
                    
                    with col3:
                        renewable_pct = esg_data['renewable_energy_pct']
                        st.metric("Renewable Energy", f"{renewable_pct:.1f}%")
                    
                    # Emissions breakdown
                    emissions_data = {
                        'Scope 1 (Direct)': esg_data['scope1_emissions'],
                        'Scope 2 (Energy)': esg_data['scope2_emissions'],
                        'Scope 3 (Other)': esg_data['scope3_emissions']
                    }
                    
                    fig_emissions = go.Figure(data=[go.Pie(
                        labels=list(emissions_data.keys()),
                        values=list(emissions_data.values()),
                        hole=0.3
                    )])
                    
                    fig_emissions.update_layout(title="Carbon Emissions Breakdown")
                    st.plotly_chart(fig_emissions, use_container_width=True)
                    
                    # Supply chain risk analysis
                    if not supply_risk.empty:
                        st.subheader("⛓️ Supply Chain Risk Analysis")
                        
                        # Risk by region
                        region_risk = supply_risk.groupby('region')['risk_score'].mean().reset_index()
                        
                        fig_risk = px.bar(
                            region_risk,
                            x='region', y='risk_score',
                            title="Average Risk Score by Region",
                            color='risk_score',
                            color_continuous_scale='RdYlGn_r'
                        )
                        
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                        # Top risk factors
                        top_risks = supply_risk.nlargest(5, 'risk_score')[['risk_factor', 'region', 'risk_score']]
                        st.subheader("🚨 Top Risk Factors")
                        st.dataframe(top_risks)
    
    elif data_source == "Company Analysis":
        st.subheader("🏢 Comprehensive Company Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.text_input("Company Name", value="Tesla Inc.")
            ticker = st.text_input("Ticker Symbol", value="TSLA")
        
        with col2:
            analysis_depth = st.selectbox("Analysis Depth", ["Basic", "Comprehensive"])
        
        if st.button("Generate Company Analysis"):
            with st.spinner("Generating comprehensive company analysis..."):
                company_score = alt_data_aggregator.create_company_alternative_score(company, ticker)
                
                if company_score:
                    # Overall score
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        composite_score = company_score['composite_score']
                        score_color = "profit" if composite_score > 70 else "loss" if composite_score < 40 else "warning-text"
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>Composite Score</strong><br>
                            <span class="{score_color}">{composite_score:.0f}/100</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        recommendation = company_score['recommendation']
                        rec_color = "profit" if recommendation == "BUY" else "loss" if recommendation == "SELL" else "warning-text"
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>Recommendation</strong><br>
                            <span class="{rec_color}">{recommendation}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        data_quality = company_score['data_quality_score']
                        st.metric("Data Quality", f"{data_quality:.0f}%")
                    
                    # Score breakdown
                    score_components = {
                        'ESG Score': company_score['esg_score'],
                        'Innovation Score': company_score['innovation_score'],
                        'Business Activity': company_score['business_activity_score'],
                        'Supply Chain Risk': company_score['supply_chain_risk_score']
                    }
                    
                    fig_components = go.Figure()
                    
                    categories = list(score_components.keys())
                    values = list(score_components.values())
                    
                    fig_components.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=company
                    ))
                    
                    fig_components.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        title="Alternative Data Score Breakdown"
                    )
                    
                    st.plotly_chart(fig_components, use_container_width=True)
                    
                    # Key insights
                    st.subheader("🔍 Key Insights")
                    
                    job_growth = company_score['job_growth_pct']
                    if job_growth > 10:
                        st.success(f"📈 Strong hiring growth: +{job_growth:.1f}% indicates business expansion")
                    elif job_growth < -10:
                        st.warning(f"📉 Declining hiring: {job_growth:.1f}% may signal contraction")
                    else:
                        st.info(f"📊 Stable hiring: {job_growth:.1f}% growth")
                    
                    esg_score = company_score['esg_score']
                    if esg_score > 70:
                        st.success(f"🌱 Strong ESG performance: {esg_score:.0f}/100")
                    elif esg_score < 40:
                        st.warning(f"⚠️ ESG concerns: {esg_score:.0f}/100")
                    else:
                        st.info(f"🔄 Moderate ESG performance: {esg_score:.0f}/100")

def backtesting_page():
    """Portfolio backtesting page"""
    st.markdown('<h2 class="section-header">📊 Portfolio Backtesting</h2>', unsafe_allow_html=True)
    
    # Initialize backtesting components
    strategy = OptionTradingStrategy()
    backtester = PortfolioBacktester()
    
    st.sidebar.markdown("### ⚙️ Backtesting Settings")
    
    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Trading Strategy",
        ["Long Call", "Long Put", "Covered Call", "Protective Put", "Iron Condor", "Straddle"]
    )
    
    # Universe selection
    universe = st.sidebar.multiselect(
        "Stock Universe",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "SPY", "QQQ"],
        default=["AAPL", "SPY"]
    )
    
    # Backtesting parameters
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
        position_size = st.slider("Position Size (%)", 1, 20, 5, 1)
    
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
        commission = st.number_input("Commission per Trade ($)", value=1.0, min_value=0.0)
        rebalance_freq = st.selectbox("Rebalancing Frequency", ["Daily", "Weekly", "Monthly"])
    
    # Strategy specific parameters
    if strategy_type in ["Long Call", "Long Put"]:
        st.sidebar.markdown("#### Strategy Parameters")
        moneyness = st.sidebar.slider("Moneyness (Strike/Spot)", 0.8, 1.2, 1.0, 0.05)
        dte_target = st.sidebar.slider("Days to Expiration", 7, 60, 30, 7)
        
    elif strategy_type == "Iron Condor":
        st.sidebar.markdown("#### Iron Condor Parameters")
        wing_width = st.sidebar.slider("Wing Width ($)", 5, 50, 10, 5)
        put_strike_pct = st.sidebar.slider("Put Strike (%)", 85, 95, 90, 1)
        call_strike_pct = st.sidebar.slider("Call Strike (%)", 105, 115, 110, 1)
    
    # Run backtest
    if st.button("Run Backtest"):
        if not universe:
            st.error("Please select at least one stock for the universe.")
            return
            
        with st.spinner("Running backtest..."):
            # Get market data for universe
            market_provider = MarketDataProvider()
            universe_data = {}
            
            for symbol in universe:
                data = market_provider.get_stock_data(symbol, period="2y")
                if not data.empty:
                    universe_data[symbol] = data
            
            if not universe_data:
                st.error("Could not fetch market data for selected universe.")
                return
            
            # Configure strategy
            strategy_config = {
                'strategy_type': strategy_type,
                'initial_capital': initial_capital,
                'position_size_pct': position_size / 100,
                'commission': commission,
                'rebalance_frequency': rebalance_freq.lower()
            }
            
            if strategy_type in ["Long Call", "Long Put"]:
                strategy_config.update({
                    'moneyness': moneyness,
                    'days_to_expiration': dte_target
                })
            elif strategy_type == "Iron Condor":
                strategy_config.update({
                    'wing_width': wing_width,
                    'put_strike_pct': put_strike_pct / 100,
                    'call_strike_pct': call_strike_pct / 100
                })
            
            # Run backtest
            backtest_results = backtester.run_backtest(
                universe_data=universe_data,
                strategy_config=strategy_config,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if backtest_results:
                display_backtest_results(backtest_results, strategy_type, universe)

def display_backtest_results(results, strategy_type, universe):
    """Display comprehensive backtest results"""
    
    # Performance summary
    st.subheader("📈 Performance Summary")
    
    performance = results['performance_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = performance['total_return']
        return_color = "profit" if total_return > 0 else "loss"
        st.markdown(f"""
        <div class="metric-card">
            <strong>Total Return</strong><br>
            <span class="{return_color}">{total_return:+.1%}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sharpe_ratio = performance['sharpe_ratio']
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col3:
        max_drawdown = performance['max_drawdown']
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    
    with col4:
        win_rate = performance['win_rate']
        st.metric("Win Rate", f"{win_rate:.1%}")
    
    # Portfolio value chart
    portfolio_history = pd.DataFrame(results['portfolio_history'])
    
    fig_portfolio = go.Figure()
    
    fig_portfolio.add_trace(go.Scatter(
        x=pd.to_datetime(portfolio_history['date']),
        y=portfolio_history['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    # Add benchmark (SPY)
    if 'benchmark_value' in portfolio_history.columns:
        fig_portfolio.add_trace(go.Scatter(
            x=pd.to_datetime(portfolio_history['date']),
            y=portfolio_history['benchmark_value'],
            mode='lines',
            name='SPY Benchmark',
            line=dict(color='red', dash='dash')
        ))
    
    fig_portfolio.update_layout(
        title=f"{strategy_type} Strategy Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_portfolio, use_container_width=True)
    
    # Returns distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'daily_returns' in results:
            daily_returns = results['daily_returns']
            
            fig_returns = go.Figure()
            fig_returns.add_histogram(
                x=daily_returns,
                bins=50,
                name="Daily Returns",
                histnorm='probability density'
            )
            
            fig_returns.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Daily Return",
                yaxis_title="Density"
            )
            
            st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        # Rolling Sharpe ratio
        if 'rolling_sharpe' in results:
            rolling_sharpe = pd.DataFrame(results['rolling_sharpe'])
            
            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Scatter(
                x=pd.to_datetime(rolling_sharpe['date']),
                y=rolling_sharpe['sharpe_ratio'],
                mode='lines',
                name='30-Day Rolling Sharpe',
                line=dict(color='green')
            ))
            
            fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                annotation_text="Good Performance")
            
            fig_sharpe.update_layout(
                title="Rolling Sharpe Ratio",
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio"
            )
            
            st.plotly_chart(fig_sharpe, use_container_width=True)
    
    # Trade analysis
    if 'trades' in results:
        st.subheader("📋 Trade Analysis")
        
        trades_df = pd.DataFrame(results['trades'])
        
        if not trades_df.empty:
            # Trade summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_trades = len(trades_df)
                st.metric("Total Trades", total_trades)
            
            with col2:
                profitable_trades = len(trades_df[trades_df['pnl'] > 0])
                st.metric("Profitable Trades", profitable_trades)
            
            with col3:
                avg_pnl = trades_df['pnl'].mean()
                st.metric("Average P&L", f"${avg_pnl:.2f}")
            
            # P&L by symbol
            if 'symbol' in trades_df.columns:
                symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().reset_index()
                
                fig_symbol_pnl = px.bar(
                    symbol_pnl,
                    x='symbol', y='pnl',
                    title="P&L by Symbol",
                    color='pnl',
                    color_continuous_scale='RdYlGn'
                )
                
                st.plotly_chart(fig_symbol_pnl, use_container_width=True)
            
            # Recent trades table
            st.subheader("Recent Trades")
            recent_trades = trades_df.tail(10)
            st.dataframe(recent_trades[['date', 'symbol', 'action', 'quantity', 'price', 'pnl']])
    
    # Risk metrics
    st.subheader("🎯 Risk Analysis")
    
    risk_metrics = results.get('risk_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        var_95 = risk_metrics.get('var_95', 0)
        st.metric("VaR (95%)", f"${var_95:,.0f}")
    
    with col2:
        cvar_95 = risk_metrics.get('cvar_95', 0)
        st.metric("CVaR (95%)", f"${cvar_95:,.0f}")
    
    with col3:
        volatility = risk_metrics.get('volatility', 0)
        st.metric("Volatility (Ann.)", f"{volatility:.1%}")
    
    with col4:
        calmar_ratio = risk_metrics.get('calmar_ratio', 0)
        st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")

def risk_management_page():
    """Risk management and portfolio analysis"""
    st.markdown('<h2 class="section-header">🎯 Risk Management</h2>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### ⚙️ Risk Analysis Settings")
    
    # Portfolio setup
    st.subheader("📊 Portfolio Construction")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Portfolio weights input
        st.write("**Portfolio Weights:**")
        
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "SPY"]
        weights = {}
        
        cols = st.columns(4)
        for i, symbol in enumerate(symbols):
            with cols[i % 4]:
                weight = st.number_input(f"{symbol} (%)", value=12.5 if i < 8 else 0, 
                                       min_value=0.0, max_value=100.0, step=0.1, key=f"weight_{symbol}")
                weights[symbol] = weight / 100
    
    with col2:
        portfolio_value = st.number_input("Portfolio Value ($)", value=1000000, min_value=1000)
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        time_horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month"], index=0)
    
    # Validate weights
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"Portfolio weights sum to {total_weight:.1%}. Please adjust to 100%.")
        return
    
    # Get market data and calculate risk metrics
    if st.button("Calculate Risk Metrics"):
        with st.spinner("Calculating portfolio risk metrics..."):
            market_provider = MarketDataProvider()
            
            # Get price data
            portfolio_data = {}
            for symbol, weight in weights.items():
                if weight > 0:
                    data = market_provider.get_stock_data(symbol, period="2y")
                    if not data.empty:
                        portfolio_data[symbol] = {
                            'prices': data['Close'],
                            'returns': data['Returns'].dropna(),
                            'weight': weight
                        }
            
            if not portfolio_data:
                st.error("Could not fetch market data for portfolio symbols.")
                return
            
            # Calculate portfolio metrics
            calculate_portfolio_risk(portfolio_data, portfolio_value, confidence_level, time_horizon)

def calculate_portfolio_risk(portfolio_data, portfolio_value, confidence_level, time_horizon):
    """Calculate and display comprehensive portfolio risk metrics"""
    
    # Prepare returns matrix
    symbols = list(portfolio_data.keys())
    returns_data = {}
    weights_array = []
    
    for symbol in symbols:
        returns_data[symbol] = portfolio_data[symbol]['returns']
        weights_array.append(portfolio_data[symbol]['weight'])
    
    returns_df = pd.DataFrame(returns_data).dropna()
    weights_array = np.array(weights_array)
    
    # Portfolio returns
    portfolio_returns = (returns_df * weights_array).sum(axis=1)
    
    # Basic risk metrics
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    portfolio_mean_return = portfolio_returns.mean() * 252
    
    # VaR calculations
    var_historical = VaRCalculator.portfolio_var(portfolio_returns, 1 - confidence_level)
    
    # Time horizon adjustment
    time_multiplier = {"1 Day": 1, "1 Week": 7, "1 Month": 30}[time_horizon]
    horizon_var = var_historical['var'] * np.sqrt(time_multiplier) * portfolio_value
    horizon_cvar = var_historical['cvar'] * np.sqrt(time_multiplier) * portfolio_value
    
    # Display risk metrics
    st.subheader("📊 Portfolio Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Volatility", f"{portfolio_vol:.1%}")
    
    with col2:
        st.metric("Expected Return", f"{portfolio_mean_return:.1%}")
    
    with col3:
        st.metric(f"VaR ({confidence_level:.0%})", f"${abs(horizon_var):,.0f}")
    
    with col4:
        st.metric(f"CVaR ({confidence_level:.0%})", f"${abs(horizon_cvar):,.0f}")
    
    # Correlation matrix
    st.subheader("🔗 Correlation Analysis")
    
    correlation_matrix = returns_df.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig_corr.update_layout(
        title="Asset Correlation Matrix",
        xaxis_title="Asset",
        yaxis_title="Asset"
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Risk decomposition
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio composition
        weights_df = pd.DataFrame({
            'Symbol': symbols,
            'Weight': [portfolio_data[s]['weight'] for s in symbols],
            'Value': [portfolio_data[s]['weight'] * portfolio_value for s in symbols]
        })
        
        fig_composition = px.pie(
            weights_df,
            values='Weight',
            names='Symbol',
            title="Portfolio Composition"
        )
        
        st.plotly_chart(fig_composition, use_container_width=True)
    
    with col2:
        # Risk contribution
        # Calculate marginal VaR for each asset
        marginal_vars = []
        portfolio_var = np.sqrt(np.dot(weights_array, np.dot(returns_df.cov() * 252, weights_array)))
        
        for i, symbol in enumerate(symbols):
            # Marginal VaR = weight * (covariance with portfolio) / portfolio variance
            cov_with_portfolio = np.dot(returns_df.cov().iloc[i] * 252, weights_array)
            marginal_var = weights_array[i] * cov_with_portfolio / (portfolio_var ** 2)
            marginal_vars.append(marginal_var)
        
        risk_contrib_df = pd.DataFrame({
            'Symbol': symbols,
            'Risk_Contribution': marginal_vars
        })
        
        fig_risk = px.bar(
            risk_contrib_df,
            x='Symbol', y='Risk_Contribution',
            title="Risk Contribution by Asset"
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Monte Carlo VaR
    st.subheader("🎲 Monte Carlo Risk Analysis")
    
    if st.button("Run Monte Carlo VaR"):
        with st.spinner("Running Monte Carlo simulation..."):
            # Monte Carlo parameters
            n_simulations = 10000
            n_days = time_multiplier
            
            # Generate random returns
            mean_returns = returns_df.mean().values
            cov_matrix = returns_df.cov().values
            
            # Simulate portfolio returns
            simulated_returns = []
            
            for _ in range(n_simulations):
                # Generate correlated random returns
                random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
                
                # Calculate portfolio returns for each day
                portfolio_daily_returns = np.dot(random_returns, weights_array)
                
                # Calculate cumulative return over horizon
                cumulative_return = np.prod(1 + portfolio_daily_returns) - 1
                simulated_returns.append(cumulative_return)
            
            simulated_returns = np.array(simulated_returns)
            
            # Calculate Monte Carlo VaR
            mc_var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            mc_cvar = np.mean(simulated_returns[simulated_returns <= mc_var])
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("MC VaR", f"${abs(mc_var * portfolio_value):,.0f}")
                st.metric("MC CVaR", f"${abs(mc_cvar * portfolio_value):,.0f}")
            
            with col2:
                # Distribution of returns
                fig_mc = go.Figure()
                fig_mc.add_histogram(
                    x=simulated_returns * 100,
                    bins=50,
                    name="Simulated Returns",
                    histnorm='probability density'
                )
                
                # Add VaR line
                fig_mc.add_vline(
                    x=mc_var * 100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"VaR ({confidence_level:.0%})"
                )
                
                fig_mc.update_layout(
                    title=f"Monte Carlo Returns Distribution ({time_horizon})",
                    xaxis_title="Portfolio Return (%)",
                    yaxis_title="Density"
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
    
    # Stress testing
    st.subheader("⚠️ Stress Testing")
    
    stress_scenarios = {
        "2008 Financial Crisis": {"market_shock": -0.30, "vol_shock": 2.0},
        "COVID-19 Market Crash": {"market_shock": -0.25, "vol_shock": 1.8},
        "Black Monday 1987": {"market_shock": -0.22, "vol_shock": 2.5},
        "Dot-com Bubble": {"market_shock": -0.35, "vol_shock": 1.5},
        "Custom Scenario": {"market_shock": 0, "vol_shock": 1.0}
    }
    
    scenario = st.selectbox("Select Stress Scenario", list(stress_scenarios.keys()))
    
    if scenario == "Custom Scenario":
        col1, col2 = st.columns(2)
        with col1:
            market_shock = st.slider("Market Shock", -0.50, 0.20, -0.20, 0.01)
        with col2:
            vol_shock = st.slider("Volatility Multiplier", 0.5, 3.0, 1.5, 0.1)
        
        stress_scenarios[scenario] = {"market_shock": market_shock, "vol_shock": vol_shock}
    
    if st.button("Run Stress Test"):
        scenario_params = stress_scenarios[scenario]
        market_shock = scenario_params["market_shock"]
        vol_shock = scenario_params["vol_shock"]
        
        # Calculate stressed portfolio return
        stressed_return = portfolio_mean_return + market_shock
        stressed_vol = portfolio_vol * vol_shock
        
        # Stressed VaR (parametric)
        from scipy.stats import norm
        stressed_var = norm.ppf(1 - confidence_level) * stressed_vol / np.sqrt(252) * np.sqrt(time_multiplier)
        stressed_portfolio_loss = stressed_var * portfolio_value
        
        # Display stress test results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Scenario", scenario)
        
        with col2:
            st.metric("Stressed VaR", f"${abs(stressed_portfolio_loss):,.0f}")
        
        with col3:
            portfolio_impact = market_shock * portfolio_value
            impact_color = "profit" if portfolio_impact > 0 else "loss"
            st.markdown(f"""
            <div class="metric-card">
                <strong>Portfolio Impact</strong><br>
                <span class="{impact_color}">${portfolio_impact:+,.0f}</span>
            </div>
            """, unsafe_allow_html=True)

def model_comparison_page():
    """Model comparison and validation page"""
    st.markdown('<h2 class="section-header">📋 Model Comparison</h2>', unsafe_allow_html=True)
    
    st.subheader("🏆 Option Pricing Model Comparison")
    
    # Model comparison setup
    col1, col2 = st.columns(2)
    
    with col1:
        # Market parameters
        S = st.number_input("Current Stock Price", value=100.0, min_value=0.01)
        K = st.number_input("Strike Price", value=100.0, min_value=0.01)
        T = st.slider("Time to Expiration (Years)", 0.01, 2.0, 0.25, 0.01)
        r = st.slider("Risk-Free Rate", 0.0, 0.10, 0.05, 0.001)
        sigma = st.slider("Volatility", 0.01, 1.0, 0.25, 0.01)
    
    with col2:
        # Model selection
        models_to_compare = st.multiselect(
            "Select Models to Compare",
            ["Black-Scholes", "Heston", "Monte Carlo", "Binomial Tree"],
            default=["Black-Scholes", "Monte Carlo"]
        )
        
        option_type = st.selectbox("Option Type", ["call", "put"])
        
        # Comparison metrics
        comparison_metrics = st.multiselect(
            "Comparison Metrics",
            ["Price", "Greeks", "Computational Time", "Accuracy"],
            default=["Price", "Greeks"]
        )
    
    if st.button("Run Model Comparison"):
        if not models_to_compare:
            st.error("Please select at least one model to compare.")
            return
        
        with st.spinner("Running model comparison..."):
            comparison_results = run_model_comparison(
                S, K, T, r, sigma, option_type, models_to_compare, comparison_metrics
            )
            
            display_model_comparison_results(comparison_results, comparison_metrics)
    
    # Market data validation
    st.subheader("📊 Market Data Validation")
    
    symbol = st.text_input("Stock Symbol for Validation", value="AAPL")
    
    if st.button("Validate Against Market Data"):
        with st.spinner("Validating models against market data..."):
            validation_results = validate_models_against_market(symbol, models_to_compare)
            
            if validation_results:
                display_validation_results(validation_results)

def run_model_comparison(S, K, T, r, sigma, option_type, models, metrics):
    """Run comprehensive model comparison"""
    import time
    
    results = {}
    
    for model_name in models:
        model_results = {}
        
        # Price calculation
        if "Price" in metrics:
            start_time = time.time()
            
            if model_name == "Black-Scholes":
                price = BlackScholesModel.option_price(S, K, T, r, sigma, option_type)
                greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, option_type)
            
            elif model_name == "Heston":
                # Use default Heston parameters
                v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.3, -0.7
                price = HestonModel.option_price_fft(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type)
                greeks = {}  # Heston Greeks calculation would be more complex
            
            elif model_name == "Monte Carlo":
                mc_engine = MonteCarloEngine(n_simulations=10000)
                S_paths = mc_engine.geometric_brownian_motion(S, r, sigma, T)
                mc_result = mc_engine.price_european_option(S_paths, K, r, T, option_type)
                price = mc_result['price']
                greeks = mc_engine.calculate_greeks_fd(S, K, T, r, sigma, option_type)
            
            elif model_name == "Binomial Tree":
                # Simplified binomial tree implementation
                n_steps = 100
                dt = T / n_steps
                u = np.exp(sigma * np.sqrt(dt))
                d = 1 / u
                p = (np.exp(r * dt) - d) / (u - d)
                
                # Build price tree
                prices = np.zeros((n_steps + 1, n_steps + 1))
                for i in range(n_steps + 1):
                    for j in range(i + 1):
                        prices[j, i] = S * (u ** (i - j)) * (d ** j)
                
                # Calculate option values
                option_values = np.zeros((n_steps + 1, n_steps + 1))
                
                # Terminal payoffs
                for j in range(n_steps + 1):
                    if option_type == "call":
                        option_values[j, n_steps] = max(0, prices[j, n_steps] - K)
                    else:
                        option_values[j, n_steps] = max(0, K - prices[j, n_steps])
                
                # Backward induction
                for i in range(n_steps - 1, -1, -1):
                    for j in range(i + 1):
                        option_values[j, i] = np.exp(-r * dt) * (
                            p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1]
                        )
                
                price = option_values[0, 0]
                greeks = {}  # Binomial Greeks would require additional calculations
            
            end_time = time.time()
            computation_time = end_time - start_time
            
            model_results['price'] = price
            model_results['computation_time'] = computation_time
            
            if "Greeks" in metrics and greeks:
                model_results['greeks'] = greeks
        
        results[model_name] = model_results
    
    return results

def display_model_comparison_results(results, metrics):
    """Display model comparison results"""
    
    # Price comparison
    if "Price" in metrics:
        st.subheader("💰 Price Comparison")
        
        price_data = []
        for model, result in results.items():
            if 'price' in result:
                price_data.append({
                    'Model': model,
                    'Price': result['price'],
                    'Computation Time (ms)': result.get('computation_time', 0) * 1000
                })
        
        if price_data:
            price_df = pd.DataFrame(price_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price comparison chart
                fig_price = px.bar(
                    price_df,
                    x='Model', y='Price',
                    title="Option Price by Model",
                    color='Price',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            with col2:
                # Computation time comparison
                fig_time = px.bar(
                    price_df,
                    x='Model', y='Computation Time (ms)',
                    title="Computation Time by Model",
                    color='Computation Time (ms)',
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Price comparison table
            st.dataframe(price_df.round(4))
    
    # Greeks comparison
    if "Greeks" in metrics:
        st.subheader("🔧 Greeks Comparison")
        
        greeks_data = []
        for model, result in results.items():
            if 'greeks' in result and result['greeks']:
                row = {'Model': model}
                row.update(result['greeks'])
                greeks_data.append(row)
        
        if greeks_data:
            greeks_df = pd.DataFrame(greeks_data)
            st.dataframe(greeks_df.round(4))
            
            # Greeks visualization
            if len(greeks_df) > 1:
                greek_names = [col for col in greeks_df.columns if col != 'Model']
                
                fig_greeks = go.Figure()
                
                for greek in greek_names:
                    if greek in greeks_df.columns:
                        fig_greeks.add_trace(go.Scatter(
                            x=greeks_df['Model'],
                            y=greeks_df[greek],
                            mode='lines+markers',
                            name=greek.capitalize()
                        ))
                
                fig_greeks.update_layout(
                    title="Greeks Comparison Across Models",
                    xaxis_title="Model",
                    yaxis_title="Greek Value"
                )
                
                st.plotly_chart(fig_greeks, use_container_width=True)

def validate_models_against_market(symbol, models):
    """Validate models against real market option prices"""
    
    # Get market data
    market_provider = MarketDataProvider()
    stock_data = market_provider.get_stock_data(symbol, period="1mo")
    options_data = market_provider.get_options_chain(symbol)
    
    if stock_data.empty or not options_data:
        st.error(f"Could not fetch market data for {symbol}")
        return None
    
    current_price = stock_data['Close'].iloc[-1]
    
    # Get risk-free rate
    risk_free_rate = market_provider.get_risk_free_rate()
    
    validation_results = {}
    
    if 'calls' in options_data and not options_data['calls'].empty:
        calls_df = options_data['calls']
        
        # Filter for liquid options (volume > 0, bid > 0)
        liquid_calls = calls_df[
            (calls_df['volume'] > 0) & 
            (calls_df['bid'] > 0) & 
            (calls_df['ask'] > 0)
        ].head(10)  # Take first 10 liquid options
        
        if not liquid_calls.empty:
            model_errors = {model: [] for model in models}
            
            for _, option in liquid_calls.iterrows():
                strike = option['strike']
                market_price = (option['bid'] + option['ask']) / 2  # Mid price
                
                # Calculate time to expiration (simplified - assumes expiration at end of trading day)
                expiry_date = pd.to_datetime(options_data['expiration_date'])
                current_date = pd.Timestamp.now()
                time_to_expiry = (expiry_date - current_date).days / 365.25
                
                if time_to_expiry > 0:
                    # Calculate implied volatility using Black-Scholes
                    try:
                        implied_vol = FinancialCalculations.implied_volatility_newton(
                            market_price, current_price, strike, time_to_expiry, risk_free_rate, 'call'
                        )
                    except:
                        implied_vol = 0.25  # Default fallback
                    
                    # Calculate model prices
                    for model in models:
                        try:
                            if model == "Black-Scholes":
                                model_price = BlackScholesModel.option_price(
                                    current_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'call'
                                )
                            elif model == "Monte Carlo":
                                mc_engine = MonteCarloEngine(n_simulations=5000)
                                S_paths = mc_engine.geometric_brownian_motion(
                                    current_price, risk_free_rate, implied_vol, time_to_expiry
                                )
                                mc_result = mc_engine.price_european_option(
                                    S_paths, strike, risk_free_rate, time_to_expiry, 'call'
                                )
                                model_price = mc_result['price']
                            else:
                                continue  # Skip other models for now
                            
                            # Calculate pricing error
                            error = abs(model_price - market_price) / market_price
                            model_errors[model].append({
                                'strike': strike,
                                'market_price': market_price,
                                'model_price': model_price,
                                'error': error,
                                'implied_vol': implied_vol
                            })
                        except:
                            continue
            
            validation_results = model_errors
    
    return validation_results

def display_validation_results(validation_results):
    """Display model validation results against market data"""
    
    st.subheader("✅ Model Validation Results")
    
    # Calculate summary statistics
    summary_stats = []
    
    for model, errors in validation_results.items():
        if errors:
            error_values = [e['error'] for e in errors]
            summary_stats.append({
                'Model': model,
                'Mean Error': np.mean(error_values),
                'Median Error': np.median(error_values),
                'Max Error': np.max(error_values),
                'RMSE': np.sqrt(np.mean([e**2 for e in error_values])),
                'Sample Size': len(errors)
            })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        
        # Display summary table
        st.dataframe(summary_df.round(4))
        
        # Error distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Mean error comparison
            fig_error = px.bar(
                summary_df,
                x='Model', y='Mean Error',
                title="Mean Pricing Error by Model",
                color='Mean Error',
                color_continuous_scale='reds_r'
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = px.bar(
                summary_df,
                x='Model', y='RMSE',
                title="Root Mean Square Error by Model",
                color='RMSE',
                color_continuous_scale='oranges_r'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Detailed error analysis
        st.subheader("📊 Detailed Error Analysis")
        
        for model, errors in validation_results.items():
            if errors:
                st.write(f"**{model} Model:**")
                
                error_df = pd.DataFrame(errors)
                
                # Scatter plot: model price vs market price
                fig_scatter = px.scatter(
                    error_df,
                    x='market_price', y='model_price',
                    title=f"{model}: Model Price vs Market Price",
                    hover_data=['strike', 'error']
                )
                
                # Add perfect prediction line
                min_price = min(error_df['market_price'].min(), error_df['model_price'].min())
                max_price = max(error_df['market_price'].max(), error_df['model_price'].max())
                
                fig_scatter.add_trace(go.Scatter(
                    x=[min_price, max_price],
                    y=[min_price, max_price],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.write("---")

# Initialize data providers (add these imports to the top)
try:
    from data.sentiment_analysis import SentimentAnalyzer, NewsDataProvider
    from visualization.charts import FinancialCharts, InteractiveCharts
    from visualization.volatility_surface import VolatilitySurfaceVisualizer
    from backtesting.strategy import OptionTradingStrategy, PortfolioBacktester
except ImportError as e:
    st.error(f"Missing module: {e}. Some features may not be available.")

# Advanced Features Pages

def exotic_options_page():
    """Exotic options laboratory"""
    st.markdown('<h2 class="section-header">🧪 Exotic Options Laboratory</h2>', unsafe_allow_html=True)
    
    exotic_engine = ExoticOptionsEngine()
    structured_products = StructuredProducts()
    
    st.sidebar.markdown("### 🎯 Exotic Option Settings")
    option_type = st.sidebar.selectbox(
        "Exotic Option Type",
        ["Barrier Options", "Asian Options", "Lookback Options", "Digital Options", "Rainbow Options", "Structured Products"]
    )
    
    # Common parameters
    S = st.sidebar.number_input("Current Price ($)", value=100.0, min_value=1.0)
    K = st.sidebar.number_input("Strike Price ($)", value=105.0, min_value=1.0)
    T = st.sidebar.number_input("Time to Expiration (years)", value=0.25, min_value=0.01, max_value=5.0)
    r = st.sidebar.number_input("Risk-free Rate", value=0.05, min_value=0.0, max_value=1.0)
    sigma = st.sidebar.number_input("Volatility", value=0.20, min_value=0.01, max_value=2.0)
    
    if option_type == "Barrier Options":
        st.subheader("🚧 Barrier Options")
        
        col1, col2 = st.columns(2)
        with col1:
            barrier = st.number_input("Barrier Level ($)", value=110.0, min_value=1.0)
            barrier_type = st.selectbox("Barrier Type", ["knock_out", "knock_in", "up_and_out", "down_and_out"])
            call_put = st.selectbox("Option Type", ["call", "put"])
        
        if st.button("Calculate Barrier Option"):
            result = exotic_engine.barrier_option_price(S, K, T, r, sigma, barrier, call_put, barrier_type)
            
            with col2:
                st.metric("Option Price", f"${result['price']:.2f}")
                st.metric("Barrier Touch Probability", f"{result.get('probability_touch', 0):.1%}")
                
                if 'greeks' in result:
                    st.write("**Greeks:**")
                    for greek, value in result['greeks'].items():
                        st.write(f"- {greek.title()}: {value:.4f}")
    
    elif option_type == "Asian Options":
        st.subheader("🌏 Asian Options (Average Price)")
        
        col1, col2 = st.columns(2)
        with col1:
            avg_type = st.selectbox("Average Type", ["arithmetic", "geometric"])
            num_obs = st.number_input("Observations", value=252, min_value=10, max_value=1000)
            call_put = st.selectbox("Option Type", ["call", "put"])
        
        if st.button("Calculate Asian Option"):
            result = exotic_engine.asian_option_price(S, K, T, r, sigma, num_obs, call_put, avg_type)
            
            with col2:
                st.metric("Option Price", f"${result['price']:.2f}")
                st.write(f"**Averaging:** {result['type']}")
                st.write(f"**Observations:** {result['averaging_observations']}")
    
    elif option_type == "Rainbow Options":
        st.subheader("🌈 Rainbow Options (Multi-Asset)")
        
        col1, col2 = st.columns(2)
        with col1:
            S2 = st.number_input("Asset 2 Price ($)", value=95.0, min_value=1.0)
            sigma2 = st.number_input("Asset 2 Volatility", value=0.25, min_value=0.01, max_value=2.0)
            correlation = st.slider("Correlation", -1.0, 1.0, 0.3)
            rainbow_type = st.selectbox("Rainbow Type", ["max", "min", "spread"])
        
        if st.button("Calculate Rainbow Option"):
            result = exotic_engine.rainbow_option_price(S, S2, K, T, r, sigma, sigma2, correlation, rainbow_type)
            
            with col2:
                st.metric("Option Price", f"${result['price']:.2f}")
                st.metric("Correlation Used", f"{result['correlation']:.2f}")
                st.write(f"**Type:** {result['type']}")
    
    elif option_type == "Structured Products":
        st.subheader("🏗️ Structured Products")
        
        product_type = st.selectbox("Product Type", ["Autocallable Note", "Reverse Convertible Note"])
        
        if product_type == "Autocallable Note":
            col1, col2 = st.columns(2)
            with col1:
                notional = st.number_input("Notional ($)", value=1000.0, min_value=100.0)
                coupon_rate = st.number_input("Coupon Rate", value=0.08, min_value=0.0, max_value=1.0)
                barrier_level = st.number_input("Barrier Level ($)", value=85.0, min_value=1.0)
                
                obs_dates = [0.25, 0.5, 0.75, 1.0]  # Quarterly observations
                
            if st.button("Price Autocallable Note"):
                result = structured_products.autocallable_note(S, notional, coupon_rate, barrier_level, obs_dates)
                
                with col2:
                    st.metric("Note Value", f"${result['value']:.2f}")
                    st.metric("Coupon PV", f"${result['coupon_pv']:.2f}")
                    st.metric("Redemption Probability", f"{result['redemption_probability']:.1%}")
                    st.metric("Yield to Call", f"{result['yield_to_call']:.1%}")

def crypto_derivatives_page():
    """Cryptocurrency derivatives trading"""
    st.markdown('<h2 class="section-header">₿ Cryptocurrency Derivatives</h2>', unsafe_allow_html=True)
    
    crypto_engine = CryptoDerivativesEngine()
    nft_derivatives = NFTDerivatives()
    
    st.sidebar.markdown("### 🎯 Crypto Settings")
    derivative_type = st.sidebar.selectbox(
        "Derivative Type",
        ["Crypto Options", "Perpetual Futures", "DeFi Options", "Yield Farming Strategy", "NFT Floor Options"]
    )
    
    if derivative_type == "Crypto Options":
        st.subheader("₿ Cryptocurrency Options")
        
        col1, col2 = st.columns(2)
        with col1:
            crypto_symbol = st.selectbox("Cryptocurrency", ["BTC", "ETH", "ADA", "SOL", "MATIC"])
            
            # Get crypto data
            crypto_data = crypto_engine.get_crypto_data(crypto_symbol, "1y")
            
            if not crypto_data.empty:
                current_price = crypto_data['Close'].iloc[-1]
                current_vol = crypto_data['Volatility_30d'].iloc[-1] if 'Volatility_30d' in crypto_data.columns else 0.8
                
                st.write(f"**Current {crypto_symbol} Price:** ${current_price:,.2f}")
                st.write(f"**30-day Volatility:** {current_vol:.1%}")
                
                K = st.number_input("Strike Price ($)", value=float(current_price * 1.05), min_value=1.0)
                T = st.number_input("Time to Expiration (years)", value=0.25, min_value=0.01, max_value=2.0)
                option_type = st.selectbox("Option Type", ["call", "put"])
                crypto_adjustments = st.checkbox("Apply Crypto Market Adjustments", value=True)
        
        if st.button("Calculate Crypto Option") and not crypto_data.empty:
            result = crypto_engine.crypto_option_price(
                current_price, K, T, 0.05, current_vol, option_type, crypto_adjustments
            )
            
            with col2:
                st.metric("Option Price", f"${result['price']:.2f}")
                if crypto_adjustments:
                    st.metric("Adjusted Volatility", f"{result['adjusted_volatility']:.1%}")
                    st.metric("Jump Component", f"${result.get('jump_component', 0):.2f}")
                    st.metric("Sentiment Impact", f"${result.get('sentiment_impact', 0):.2f}")
                
                st.write("**Greeks:**")
                for greek, value in result.get('greeks', {}).items():
                    st.write(f"- {greek.title()}: {value:.4f}")
    
    elif derivative_type == "Perpetual Futures":
        st.subheader("⚡ Perpetual Futures")
        
        col1, col2 = st.columns(2)
        with col1:
            crypto_symbol = st.selectbox("Cryptocurrency", ["BTC", "ETH", "ADA", "SOL"])
            crypto_data = crypto_engine.get_crypto_data(crypto_symbol, "1y")
            
            if not crypto_data.empty:
                current_price = crypto_data['Close'].iloc[-1]
                st.write(f"**Spot Price:** ${current_price:,.2f}")
                
                funding_rate = st.number_input("8h Funding Rate (%)", value=0.01, min_value=-0.5, max_value=0.5) / 100
                premium_index = st.number_input("Premium/Discount (%)", value=0.0, min_value=-5.0, max_value=5.0) / 100
        
        if st.button("Calculate Perpetual Futures") and not crypto_data.empty:
            result = crypto_engine.perpetual_futures_price(current_price, funding_rate, premium_index)
            
            with col2:
                st.metric("Mark Price", f"${result['mark_price']:,.2f}")
                st.metric("Basis", f"${result['basis']:,.2f}")
                st.metric("Annualized Basis", f"{result['annualized_basis']:.1%}")
                st.metric("Funding Payment", f"${result['funding_payment']:,.2f}")
                st.metric("Time to Funding", f"{result['time_to_funding']:.1f} hours")

def ai_enhanced_models_page():
    """AI-Enhanced financial models"""
    st.markdown('<h2 class="section-header">🔮 AI-Enhanced Financial Models</h2>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🎯 AI Model Settings")
    ai_model_type = st.sidebar.selectbox(
        "AI Model Type",
        ["Reinforcement Learning Trader", "Transformer Price Predictor", "AutoML Model Selection"]
    )
    
    if ai_model_type == "Reinforcement Learning Trader":
        st.subheader("🤖 Reinforcement Learning Trading Agent")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Stock Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"])
            training_episodes = st.number_input("Training Episodes", value=500, min_value=100, max_value=2000)
            lookback_window = st.number_input("Lookback Window", value=20, min_value=5, max_value=60)
            
            # Get market data
            market_provider = MarketDataProvider()
            data = market_provider.get_stock_data(symbol, period='2y', interval='1d')
        
        if st.button("Train RL Agent") and not data.empty:
            rl_trader = ReinforcementLearningTrader()
            
            with st.spinner("Training reinforcement learning agent..."):
                results = rl_trader.train_rl_agent(data, lookback_window, training_episodes)
            
            with col2:
                if 'error' not in results:
                    st.metric("Training Episodes", results['trained_episodes'])
                    st.metric("Average Reward", f"{results['average_reward']:.2f}")
                    st.metric("Win Rate", f"{results['win_rate']:.1%}")
                    st.metric("Q-table Size", results['q_table_size'])
                    st.metric("Convergence Score", f"{results.get('convergence_score', 0):.3f}")
                    
                    # Plot training progress
                    if 'episode_rewards' in results:
                        rewards_df = pd.DataFrame({
                            'Episode': range(len(results['episode_rewards'])),
                            'Reward': results['episode_rewards']
                        })
                        
                        fig = px.line(rewards_df, x='Episode', y='Reward', 
                                    title="RL Training Progress")
                        st.plotly_chart(fig, use_container_width=True)
    
    elif ai_model_type == "Transformer Price Predictor":
        st.subheader("🧠 Transformer-Based Price Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Stock Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"])
            sequence_length = st.number_input("Sequence Length", value=60, min_value=20, max_value=120)
            
            # Get market data
            market_provider = MarketDataProvider()
            data = market_provider.get_stock_data(symbol, period='2y', interval='1d')
        
        if st.button("Run Transformer Analysis") and not data.empty:
            transformer = TransformerPricePredictor()
            
            with st.spinner("Creating transformer features and running attention analysis..."):
                sequences = transformer.create_transformer_features(data)
                
                if len(sequences) > 0:
                    attention_results = transformer.attention_mechanism_simulation(sequences)
            
            with col2:
                if len(sequences) > 0 and 'error' not in attention_results:
                    st.metric("Sequences Generated", len(sequences))
                    st.metric("Feature Dimensions", sequences.shape[2] if len(sequences.shape) > 2 else 0)
                    st.metric("Predictions Made", len(attention_results.get('predictions', [])))
                    
                    # Feature importance
                    if 'feature_importance' in attention_results:
                        importance_df = pd.DataFrame({
                            'Feature': ['Returns', 'High_Change', 'Low_Change', 'Volume_Ratio', 
                                      'RSI', 'MACD', 'BB_Position', 'Volatility', 'Skewness', 'Kurtosis'],
                            'Importance': attention_results['feature_importance']
                        })
                        
                        fig = px.bar(importance_df, x='Feature', y='Importance', 
                                   title="Transformer Feature Importance")
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Temporal attention weights
                    if 'temporal_importance' in attention_results:
                        temporal_df = pd.DataFrame({
                            'Time_Step': range(len(attention_results['temporal_importance'])),
                            'Attention_Weight': attention_results['temporal_importance']
                        })
                        
                        fig = px.line(temporal_df, x='Time_Step', y='Attention_Weight',
                                    title="Temporal Attention Weights")
                        st.plotly_chart(fig, use_container_width=True)

def real_time_risk_page():
    """Real-time risk management dashboard"""
    st.markdown('<h2 class="section-header">🎯 Real-Time Risk Management Engine</h2>', unsafe_allow_html=True)
    
    risk_engine = RealTimeRiskEngine()
    
    st.sidebar.markdown("### 🎯 Risk Settings")
    
    # Portfolio simulation inputs
    symbols = st.sidebar.multiselect(
        "Portfolio Symbols",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "NVDA", "META"],
        default=["AAPL", "SPY"]
    )
    
    portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=100000, min_value=1000)
    confidence_level = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95)
    
    if symbols:
        # Create portfolio weights (equal weight for simplicity)
        weights = [1.0 / len(symbols)] * len(symbols)
        values = [portfolio_value * w for w in weights]
        
        portfolio = {
            'symbols': symbols,
            'weights': weights,
            'values': values
        }
        
        # Get market data for risk analysis
        market_provider = MarketDataProvider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Risk Dashboard")
            
            # Use first symbol's data for demonstration
            market_data = market_provider.get_stock_data(symbols[0], period='1y', interval='1d')
            
            if not market_data.empty:
                # Real-time risk monitoring
                risk_results = risk_engine.real_time_risk_monitor(
                    portfolio, market_data, confidence_level
                )
                
                # Risk metrics display
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Portfolio VaR (1d)", f"${risk_results['var_1d']:,.0f}")
                    st.metric("VaR Percentage", f"{risk_results['var_percentage']:.2f}%")
                
                with metric_cols[1]:
                    st.metric("Portfolio VaR (10d)", f"${risk_results['var_10d']:,.0f}")
                    st.metric("Overall Risk Score", f"{risk_results['overall_risk_score']:.0f}/100")
                
                with metric_cols[2]:
                    st.metric("Liquidity Score", f"{risk_results['liquidity_score']:.1f}")
                    st.metric("Concentration Score", f"{risk_results['concentration_score']:.1f}")
                
                with metric_cols[3]:
                    st.metric("Market Regime", risk_results['market_regime']['regime'])
                    st.metric("Regime Confidence", f"{risk_results['market_regime']['confidence']:.1%}")
                
                # Risk alerts
                if risk_results['risk_alerts']:
                    st.subheader("🚨 Risk Alerts")
                    for alert in risk_results['risk_alerts']:
                        if alert['severity'] == 'CRITICAL':
                            st.error(f"**{alert['type']}**: {alert['message']}")
                        elif alert['severity'] == 'WARNING':
                            st.warning(f"**{alert['type']}**: {alert['message']}")
                        else:
                            st.info(f"**{alert['type']}**: {alert['message']}")
                
                # Stress test results
                st.subheader("🧪 Stress Test Results")
                stress_data = []
                for scenario, result in risk_results['stress_test_results'].items():
                    stress_data.append({
                        'Scenario': scenario.replace('_', ' ').title(),
                        'Loss ($)': result['loss'],
                        'Loss (%)': result['loss_percentage'],
                        'New Portfolio Value ($)': result['new_portfolio_value']
                    })
                
                stress_df = pd.DataFrame(stress_data)
                st.dataframe(stress_df, use_container_width=True)
                
                # Stress test visualization
                fig = px.bar(stress_df, x='Scenario', y='Loss (%)', 
                           title="Stress Test Loss Scenarios",
                           color='Loss (%)', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hedge recommendations
            st.subheader("💡 Hedge Recommendations")
            
            for rec in risk_results['hedge_recommendations']:
                priority_color = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠', 
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(rec['priority'], '⚪')
                
                st.write(f"{priority_color} **{rec['type']}**")
                st.write(f"Action: {rec['action']}")
                st.write(f"Rationale: {rec['rationale']}")
                st.write(f"Allocation: {rec['allocation']:.1%}")
                st.write("---")
            
            # Dynamic position sizing
            st.subheader("📏 Dynamic Position Sizing")
            
            symbol_for_sizing = st.selectbox("Symbol for Position Sizing", symbols)
            target_risk = st.slider("Target Risk (%)", 1.0, 5.0, 2.0) / 100
            
            if symbol_for_sizing:
                # Get current volatility
                symbol_data = market_provider.get_stock_data(symbol_for_sizing, period='3m', interval='1d')
                if not symbol_data.empty:
                    current_vol = symbol_data['Close'].pct_change().std() * np.sqrt(252)
                    
                    sizing_result = risk_engine.dynamic_position_sizing(
                        symbol_for_sizing, current_vol, target_risk, portfolio_value
                    )
                    
                    st.metric("Optimal Position Size", f"${sizing_result['optimal_position_size']:,.0f}")
                    st.metric("Position %", f"{sizing_result['position_percentage']:.1f}%")
                    st.metric("Kelly Fraction", f"{sizing_result['kelly_fraction']:.3f}")
                    st.write(f"**Recommendation:** {sizing_result['recommendation']}")

def quantum_optimizer_page():
    """Quantum-inspired portfolio optimization"""
    st.markdown('<h2 class="section-header">⚡ Quantum-Inspired Portfolio Optimizer</h2>', unsafe_allow_html=True)
    
    quantum_optimizer = QuantumInspiredOptimizer()
    
    st.sidebar.markdown("### 🎯 Quantum Settings")
    
    # Portfolio universe
    universe_symbols = st.sidebar.multiselect(
        "Investment Universe",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY", "QQQ", "VTI"],
        default=["AAPL", "GOOGL", "MSFT", "SPY"]
    )
    
    risk_tolerance = st.sidebar.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.1)
    quantum_iterations = st.sidebar.number_input("Quantum Iterations", value=1000, min_value=100, max_value=5000)
    
    if universe_symbols and len(universe_symbols) >= 2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🌌 Quantum Portfolio Optimization")
            
            # Get return data for optimization
            market_provider = MarketDataProvider()
            returns_data = {}
            
            for symbol in universe_symbols:
                data = market_provider.get_stock_data(symbol, period='2y', interval='1d')
                if not data.empty:
                    returns_data[symbol] = data['Close'].pct_change().dropna()
            
            if len(returns_data) >= 2:
                # Align all return series
                returns_df = pd.DataFrame(returns_data).dropna()
                
                if st.button("🚀 Run Quantum Optimization"):
                    with st.spinner("Running quantum-inspired optimization..."):
                        results = quantum_optimizer.quantum_portfolio_optimization(
                            returns_df, risk_tolerance, quantum_iterations
                        )
                    
                    if 'error' not in results:
                        # Display optimization results
                        st.subheader("📊 Optimization Results")
                        
                        # Metrics
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Expected Return", f"{results['expected_return']:.1%}")
                        with metric_cols[1]:
                            st.metric("Expected Risk", f"{results['expected_risk']:.1%}")
                        with metric_cols[2]:
                            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                        with metric_cols[3]:
                            st.metric("Utility Score", f"{results['utility_score']:.3f}")
                        
                        # Portfolio weights
                        weights_df = pd.DataFrame({
                            'Asset': results['assets'],
                            'Weight': results['optimal_weights'],
                            'Weight_Pct': [w * 100 for w in results['optimal_weights']]
                        })
                        
                        # Portfolio allocation pie chart
                        fig_pie = px.pie(weights_df, values='Weight_Pct', names='Asset',
                                        title="Quantum-Optimized Portfolio Allocation")
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Weights bar chart
                        fig_bar = px.bar(weights_df, x='Asset', y='Weight_Pct',
                                        title="Portfolio Weights (%)",
                                        color='Weight_Pct', color_continuous_scale='viridis')
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Display weights table
                        st.subheader("📋 Portfolio Weights")
                        weights_display = weights_df.copy()
                        weights_display['Weight_Pct'] = weights_display['Weight_Pct'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(weights_display[['Asset', 'Weight_Pct']], use_container_width=True)
        
        with col2:
            st.subheader("🔬 Quantum Algorithm Info")
            
            st.info("""
            **Quantum-Inspired Optimization Features:**
            
            🌀 **Superposition**: Portfolio weights initialized in quantum superposition state
            
            ❄️ **Annealing**: Simulated quantum annealing with temperature cooling
            
            🎯 **Measurement**: Quantum state collapse to classical portfolio weights
            
            ⚡ **Entanglement**: Correlated weight updates across assets
            
            🔄 **Iteration**: Quantum state evolution over multiple cycles
            """)
            
            # Risk tolerance explanation
            st.subheader("🎚️ Risk Tolerance Guide")
            
            if risk_tolerance < 0.3:
                st.write("🛡️ **Conservative**: Focus on risk minimization")
            elif risk_tolerance < 0.7:
                st.write("⚖️ **Balanced**: Equal weight to risk and return")
            else:
                st.write("🚀 **Aggressive**: Focus on return maximization")
            
            # Algorithm parameters
            st.subheader("⚙️ Algorithm Parameters")
            st.write(f"**Risk Tolerance:** {risk_tolerance:.1f}")
            st.write(f"**Quantum Iterations:** {quantum_iterations:,}")
            st.write(f"**Universe Size:** {len(universe_symbols)} assets")
            st.write(f"**Data Period:** 2 years daily returns")

if __name__ == "__main__":
    main()
