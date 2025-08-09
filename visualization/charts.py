import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import streamlit as st

class FinancialCharts:
    """Advanced financial visualization charts"""
    
    def __init__(self):
        self.color_palette = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#42a5f5',
            'volume': '#ffa726',
            'background': '#1e1e1e',
            'grid': '#333333',
            'text': '#ffffff'
        }
    
    def create_candlestick_chart(self, data: pd.DataFrame, title: str = "Stock Price") -> go.Figure:
        """Create interactive candlestick chart with volume"""
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_width=[0.2, 0.7],
            subplot_titles=[title, "Volume"]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing_line_color=self.color_palette['bullish'],
                decreasing_line_color=self.color_palette['bearish']
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add moving averages if available
        if 'MA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_20'],
                    mode='lines',
                    name='20-day MA',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_50'],
                    mode='lines',
                    name='50-day MA',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True,
            template="plotly_dark"
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str, 
                         title: str = "Line Chart") -> go.Figure:
        """Create interactive line chart"""
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(color=self.color_palette['neutral'], width=2),
                marker=dict(size=4)
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_dark",
            hovermode='x unified'
        )
        
        return fig
    
    def create_returns_distribution(self, returns: pd.Series, title: str = "Returns Distribution") -> go.Figure:
        """Create returns distribution histogram with statistics"""
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name="Returns",
                opacity=0.7,
                marker_color=self.color_palette['neutral']
            )
        )
        
        # Add normal distribution overlay
        mean_return = returns.mean()
        std_return = returns.std()
        
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1/(std_return * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)
        
        # Scale normal distribution to match histogram
        hist_counts, _ = np.histogram(returns, bins=50)
        scale_factor = hist_counts.max() / normal_dist.max()
        normal_dist_scaled = normal_dist * scale_factor
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist_scaled,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', dash='dash', width=2)
            )
        )
        
        # Add vertical lines for key statistics
        fig.add_vline(x=mean_return, line_dash="dash", line_color="green",
                     annotation_text=f"Mean: {mean_return:.4f}")
        
        # Add VaR lines
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        fig.add_vline(x=var_95, line_dash="dot", line_color="orange",
                     annotation_text=f"VaR 95%: {var_95:.4f}")
        fig.add_vline(x=var_99, line_dash="dot", line_color="red",
                     annotation_text=f"VaR 99%: {var_99:.4f}")
        
        fig.update_layout(
            title=title,
            xaxis_title="Returns",
            yaxis_title="Frequency",
            template="plotly_dark",
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                  title: str = "Correlation Matrix") -> go.Figure:
        """Create interactive correlation heatmap"""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            width=600,
            height=600
        )
        
        return fig
    
    def create_portfolio_performance(self, portfolio_data: pd.DataFrame, 
                                   benchmark_data: Optional[pd.DataFrame] = None,
                                   title: str = "Portfolio Performance") -> go.Figure:
        """Create portfolio performance chart with benchmark comparison"""
        
        fig = go.Figure()
        
        # Portfolio performance
        fig.add_trace(
            go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['cumulative_returns'],
                mode='lines',
                name='Portfolio',
                line=dict(color=self.color_palette['bullish'], width=3)
            )
        )
        
        # Benchmark comparison if provided
        if benchmark_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_data['cumulative_returns'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.color_palette['bearish'], width=2, dash='dash')
                )
            )
        
        # Add drawdown chart as secondary y-axis
        if 'drawdown' in portfolio_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_data.index,
                    y=portfolio_data['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    yaxis='y2'
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            yaxis2=dict(
                title="Drawdown",
                overlaying='y',
                side='right',
                range=[portfolio_data['drawdown'].min() if 'drawdown' in portfolio_data.columns else -0.5, 0]
            ),
            template="plotly_dark",
            hovermode='x unified'
        )
        
        return fig
    
    def create_option_payoff_diagram(self, spot_prices: np.ndarray, payoffs: np.ndarray,
                                   strike: float, option_type: str, 
                                   title: str = "Option Payoff Diagram") -> go.Figure:
        """Create option payoff diagram"""
        
        fig = go.Figure()
        
        # Payoff line
        color = self.color_palette['bullish'] if option_type == 'call' else self.color_palette['bearish']
        
        fig.add_trace(
            go.Scatter(
                x=spot_prices,
                y=payoffs,
                mode='lines',
                name=f'{option_type.title()} Payoff',
                line=dict(color=color, width=3)
            )
        )
        
        # Break-even line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Break-even")
        
        # Strike price line
        fig.add_vline(x=strike, line_dash="dot", line_color="orange",
                     annotation_text=f"Strike: {strike}")
        
        # Fill profitable area
        profitable_mask = payoffs > 0
        if profitable_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=spot_prices[profitable_mask],
                    y=payoffs[profitable_mask],
                    fill='tonexty',
                    fillcolor=f'rgba({color[1:3]}, {color[3:5]}, {color[5:7]}, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Profit Zone',
                    showlegend=False
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Spot Price at Expiration",
            yaxis_title="Profit/Loss",
            template="plotly_dark",
            hovermode='x'
        )
        
        return fig
    
    def create_volatility_cone(self, historical_vols: Dict[str, pd.Series],
                              title: str = "Volatility Cone") -> go.Figure:
        """Create volatility cone chart showing percentiles"""
        
        fig = go.Figure()
        
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'blue', 'orange', 'red']
        
        for i, (period, vol_series) in enumerate(historical_vols.items()):
            vol_percentiles = [np.percentile(vol_series, p) for p in percentiles]
            
            for j, (percentile, vol_val, color) in enumerate(zip(percentiles, vol_percentiles, colors)):
                fig.add_trace(
                    go.Scatter(
                        x=[period],
                        y=[vol_val],
                        mode='markers',
                        name=f'{percentile}th Percentile' if i == 0 else None,
                        marker=dict(color=color, size=8),
                        showlegend=(i == 0),
                        legendgroup=f'p{percentile}'
                    )
                )
        
        # Connect percentiles with lines
        periods = list(historical_vols.keys())
        for j, percentile in enumerate(percentiles):
            percentile_values = []
            for period in periods:
                vol_series = historical_vols[period]
                percentile_values.append(np.percentile(vol_series, percentile))
            
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=percentile_values,
                    mode='lines',
                    name=f'{percentile}th Percentile Line',
                    line=dict(color=colors[j], dash='dash'),
                    showlegend=False
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Volatility",
            template="plotly_dark"
        )
        
        return fig

class InteractiveCharts:
    """Interactive financial charts with advanced features"""
    
    def __init__(self):
        self.charts = FinancialCharts()
    
    def create_multi_asset_comparison(self, asset_data: Dict[str, pd.DataFrame],
                                    normalize: bool = True,
                                    title: str = "Multi-Asset Performance") -> go.Figure:
        """Create multi-asset performance comparison chart"""
        
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (asset_name, data) in enumerate(asset_data.items()):
            if 'Close' in data.columns:
                prices = data['Close']
                
                if normalize:
                    # Normalize to start at 100
                    normalized_prices = (prices / prices.iloc[0]) * 100
                    y_data = normalized_prices
                    y_title = "Normalized Price (Base = 100)"
                else:
                    y_data = prices
                    y_title = "Price"
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=y_data,
                        mode='lines',
                        name=asset_name,
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_title,
            template="plotly_dark",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_risk_return_scatter(self, assets_data: Dict[str, Dict[str, float]],
                                  title: str = "Risk-Return Analysis") -> go.Figure:
        """Create risk-return scatter plot"""
        
        fig = go.Figure()
        
        assets = list(assets_data.keys())
        returns = [assets_data[asset]['return'] for asset in assets]
        risks = [assets_data[asset]['risk'] for asset in assets]
        sharpe_ratios = [assets_data[asset].get('sharpe', 0) for asset in assets]
        
        # Color by Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns,
                mode='markers+text',
                text=assets,
                textposition="top center",
                marker=dict(
                    size=12,
                    color=sharpe_ratios,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Sharpe Ratio"),
                    line=dict(width=1, color='white')
                ),
                name="Assets"
            )
        )
        
        # Add efficient frontier if data available
        if len(assets) > 2:
            # Simulate efficient frontier
            frontier_risks = np.linspace(min(risks), max(risks), 50)
            frontier_returns = []
            
            for risk in frontier_risks:
                # Simplified efficient frontier calculation
                max_return = max(returns)
                min_risk = min(risks)
                expected_return = min_return + (risk - min_risk) * (max_return - min(returns)) / (max(risks) - min_risk)
                frontier_returns.append(expected_return)
            
            fig.add_trace(
                go.Scatter(
                    x=frontier_risks,
                    y=frontier_returns,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='cyan', dash='dash', width=2)
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            template="plotly_dark",
            showlegend=True
        )
        
        return fig
    
    def create_monte_carlo_paths(self, simulation_data: np.ndarray, 
                               percentiles: List[int] = [5, 25, 50, 75, 95],
                               title: str = "Monte Carlo Simulation Paths") -> go.Figure:
        """Create Monte Carlo simulation paths visualization"""
        
        fig = go.Figure()
        
        n_simulations, n_steps = simulation_data.shape
        time_steps = np.arange(n_steps)
        
        # Add sample paths (subset for performance)
        sample_size = min(100, n_simulations)
        sample_indices = np.random.choice(n_simulations, sample_size, replace=False)
        
        for i in sample_indices[:20]:  # Show only first 20 for clarity
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=simulation_data[i, :],
                    mode='lines',
                    line=dict(color='lightblue', width=0.5),
                    opacity=0.3,
                    showlegend=False
                )
            )
        
        # Add percentile bands
        percentile_colors = ['red', 'orange', 'green', 'orange', 'red']
        
        for i, (percentile, color) in enumerate(zip(percentiles, percentile_colors)):
            percentile_values = np.percentile(simulation_data, percentile, axis=0)
            
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=percentile_values,
                    mode='lines',
                    name=f'{percentile}th Percentile',
                    line=dict(color=color, width=2)
                )
            )
        
        # Fill area between percentiles
        p25_values = np.percentile(simulation_data, 25, axis=0)
        p75_values = np.percentile(simulation_data, 75, axis=0)
        
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_steps, time_steps[::-1]]),
                y=np.concatenate([p25_values, p75_values[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='25th-75th Percentile',
                showlegend=False
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Steps",
            yaxis_title="Value",
            template="plotly_dark",
            hovermode='x'
        )
        
        return fig
    
    def create_greeks_heatmap(self, greeks_data: pd.DataFrame,
                             title: str = "Options Greeks Heatmap") -> go.Figure:
        """Create heatmap visualization for options Greeks"""
        
        # Prepare data for heatmap
        greek_names = ['delta', 'gamma', 'theta', 'vega', 'rho']
        available_greeks = [g for g in greek_names if g in greeks_data.columns]
        
        if not available_greeks:
            st.error("No Greeks data available for heatmap")
            return go.Figure()
        
        # Normalize Greeks for better visualization
        normalized_data = greeks_data[available_greeks].copy()
        for greek in available_greeks:
            max_val = abs(normalized_data[greek]).max()
            if max_val > 0:
                normalized_data[greek] = normalized_data[greek] / max_val
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized_data.values.T,
            x=greeks_data.index,
            y=available_greeks,
            colorscale='RdBu',
            zmid=0,
            hoverongaps=False,
            text=np.round(greeks_data[available_greeks].values.T, 4),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Strike Price",
            yaxis_title="Greeks",
            template="plotly_dark"
        )
        
        return fig
    
    def create_sentiment_gauge(self, sentiment_score: float, 
                              title: str = "Market Sentiment") -> go.Figure:
        """Create sentiment gauge chart"""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score * 100,  # Convert to 0-100 scale
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 40], 'color': "orange"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=400
        )
        
        return fig
