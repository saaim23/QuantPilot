import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize

class VolatilitySurfaceVisualizer:
    """Advanced volatility surface visualization and analysis"""
    
    def __init__(self):
        self.color_scales = {
            'volatility': 'Viridis',
            'diverging': 'RdBu',
            'temperature': 'Hot',
            'cool': 'Blues'
        }
    
    def create_3d_surface(self, surface_data: pd.DataFrame, 
                         title: str = "Implied Volatility Surface") -> go.Figure:
        """Create 3D implied volatility surface"""
        
        try:
            # Prepare data for 3D surface
            strikes = sorted(surface_data['Strike'].unique())
            expiries = sorted(surface_data['Expiry'].unique())
            
            # Create meshgrid
            strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
            
            # Interpolate volatility values
            vol_grid = np.zeros_like(strike_grid)
            
            for i, expiry in enumerate(expiries):
                for j, strike in enumerate(strikes):
                    # Find closest point in data
                    subset = surface_data[
                        (abs(surface_data['Expiry'] - expiry) < 0.01) & 
                        (abs(surface_data['Strike'] - strike) < 0.01)
                    ]
                    
                    if not subset.empty:
                        vol_grid[i, j] = subset['ImpliedVol'].iloc[0]
                    else:
                        # Interpolate if exact match not found
                        nearby = surface_data[
                            (abs(surface_data['Expiry'] - expiry) < 0.1) & 
                            (abs(surface_data['Strike'] - strike) < strike * 0.1)
                        ]
                        if not nearby.empty:
                            vol_grid[i, j] = nearby['ImpliedVol'].mean()
                        else:
                            vol_grid[i, j] = surface_data['ImpliedVol'].mean()
            
            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(
                z=vol_grid,
                x=strike_grid,
                y=expiry_grid,
                colorscale=self.color_scales['volatility'],
                colorbar=dict(title="Implied Volatility"),
                hovertemplate='Strike: %{x}<br>Expiry: %{y}<br>Vol: %{z:.3f}<extra></extra>'
            )])
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Strike Price',
                    yaxis_title='Time to Expiry (Years)',
                    zaxis_title='Implied Volatility',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                template="plotly_dark",
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating 3D surface: {str(e)}")
            return go.Figure()
    
    def create_volatility_smile(self, surface_data: pd.DataFrame, 
                               selected_expiry: float,
                               title: str = "Volatility Smile") -> go.Figure:
        """Create volatility smile for specific expiry"""
        
        # Filter data for selected expiry
        smile_data = surface_data[
            abs(surface_data['Expiry'] - selected_expiry) < 0.01
        ].copy()
        
        if smile_data.empty:
            st.warning(f"No data available for expiry {selected_expiry}")
            return go.Figure()
        
        # Sort by moneyness
        smile_data = smile_data.sort_values('Moneyness')
        
        fig = go.Figure()
        
        # Volatility smile
        fig.add_trace(
            go.Scatter(
                x=smile_data['Moneyness'],
                y=smile_data['ImpliedVol'],
                mode='lines+markers',
                name=f'Implied Vol (T={selected_expiry:.2f}Y)',
                line=dict(color='blue', width=3),
                marker=dict(size=8, color='lightblue', line=dict(width=1, color='blue'))
            )
        )
        
        # Add ATM line
        fig.add_vline(
            x=1.0, 
            line_dash="dash", 
            line_color="red",
            annotation_text="ATM"
        )
        
        # Fit polynomial for smooth curve
        if len(smile_data) > 3:
            try:
                coeffs = np.polyfit(smile_data['Moneyness'], smile_data['ImpliedVol'], 3)
                x_smooth = np.linspace(smile_data['Moneyness'].min(), 
                                     smile_data['Moneyness'].max(), 100)
                y_smooth = np.polyval(coeffs, x_smooth)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_smooth,
                        y=y_smooth,
                        mode='lines',
                        name='Fitted Curve',
                        line=dict(color='red', dash='dash', width=2)
                    )
                )
            except:
                pass
        
        fig.update_layout(
            title=title,
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Implied Volatility',
            template="plotly_dark",
            hovermode='x'
        )
        
        return fig
    
    def create_term_structure(self, surface_data: pd.DataFrame,
                             moneyness_level: float = 1.0,
                             title: str = "Volatility Term Structure") -> go.Figure:
        """Create volatility term structure for specific moneyness"""
        
        # Filter data for ATM or specified moneyness level
        tolerance = 0.05
        term_data = surface_data[
            abs(surface_data['Moneyness'] - moneyness_level) < tolerance
        ].copy()
        
        if term_data.empty:
            st.warning(f"No data available for moneyness {moneyness_level}")
            return go.Figure()
        
        # Sort by expiry
        term_data = term_data.sort_values('Expiry')
        
        fig = go.Figure()
        
        # Term structure
        fig.add_trace(
            go.Scatter(
                x=term_data['Expiry'],
                y=term_data['ImpliedVol'],
                mode='lines+markers',
                name=f'Term Structure (K/S={moneyness_level:.2f})',
                line=dict(color='green', width=3),
                marker=dict(size=8, color='lightgreen', line=dict(width=1, color='green'))
            )
        )
        
        # Add trend line
        if len(term_data) > 2:
            try:
                coeffs = np.polyfit(term_data['Expiry'], term_data['ImpliedVol'], 1)
                x_trend = np.array([term_data['Expiry'].min(), term_data['Expiry'].max()])
                y_trend = np.polyval(coeffs, x_trend)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name='Trend',
                        line=dict(color='orange', dash='dash', width=2)
                    )
                )
                
                # Add trend annotation
                slope = coeffs[0]
                trend_text = f"Slope: {slope:.4f}"
                fig.add_annotation(
                    x=term_data['Expiry'].mean(),
                    y=term_data['ImpliedVol'].max(),
                    text=trend_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="orange"
                )
            except:
                pass
        
        fig.update_layout(
            title=title,
            xaxis_title='Time to Expiry (Years)',
            yaxis_title='Implied Volatility',
            template="plotly_dark",
            hovermode='x'
        )
        
        return fig
    
    def create_skew_analysis(self, surface_data: pd.DataFrame,
                            title: str = "Volatility Skew Analysis") -> go.Figure:
        """Analyze volatility skew across different expiries"""
        
        expiries = sorted(surface_data['Expiry'].unique())
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, expiry in enumerate(expiries[:6]):  # Limit to 6 expiries for clarity
            expiry_data = surface_data[
                abs(surface_data['Expiry'] - expiry) < 0.01
            ].sort_values('Moneyness')
            
            if len(expiry_data) > 2:
                fig.add_trace(
                    go.Scatter(
                        x=expiry_data['Moneyness'],
                        y=expiry_data['ImpliedVol'],
                        mode='lines+markers',
                        name=f'{expiry:.2f}Y',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6)
                    )
                )
        
        # Add ATM reference line
        fig.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="ATM"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Implied Volatility',
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
    
    def create_volatility_heatmap(self, surface_data: pd.DataFrame,
                                 title: str = "Volatility Heatmap") -> go.Figure:
        """Create 2D heatmap of volatility surface"""
        
        # Pivot data for heatmap
        pivot_data = surface_data.pivot_table(
            values='ImpliedVol',
            index='Expiry',
            columns='Strike',
            aggfunc='mean'
        )
        
        if pivot_data.empty:
            st.error("Cannot create heatmap: insufficient data")
            return go.Figure()
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=self.color_scales['volatility'],
            colorbar=dict(title="Implied Volatility"),
            hoverongaps=False,
            text=np.round(pivot_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Strike Price',
            yaxis_title='Time to Expiry (Years)',
            template="plotly_dark"
        )
        
        return fig
    
    def analyze_volatility_surface_metrics(self, surface_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various volatility surface metrics"""
        
        try:
            metrics = {}
            
            # ATM volatility by expiry
            atm_vols = []
            for expiry in surface_data['Expiry'].unique():
                expiry_data = surface_data[
                    abs(surface_data['Expiry'] - expiry) < 0.01
                ]
                # Find closest to ATM
                atm_data = expiry_data.loc[
                    abs(expiry_data['Moneyness'] - 1.0).idxmin()
                ]
                atm_vols.append({
                    'expiry': expiry,
                    'atm_vol': atm_data['ImpliedVol']
                })
            
            metrics['atm_term_structure'] = atm_vols
            
            # Skew metrics for each expiry
            skew_metrics = []
            for expiry in surface_data['Expiry'].unique():
                expiry_data = surface_data[
                    abs(surface_data['Expiry'] - expiry) < 0.01
                ].sort_values('Moneyness')
                
                if len(expiry_data) > 2:
                    # Calculate skew (slope)
                    skew = np.polyfit(expiry_data['Moneyness'], expiry_data['ImpliedVol'], 1)[0]
                    
                    # Calculate convexity (second derivative approximation)
                    if len(expiry_data) > 3:
                        convexity = np.polyfit(expiry_data['Moneyness'], expiry_data['ImpliedVol'], 2)[0]
                    else:
                        convexity = 0
                    
                    skew_metrics.append({
                        'expiry': expiry,
                        'skew': skew,
                        'convexity': convexity,
                        'vol_range': expiry_data['ImpliedVol'].max() - expiry_data['ImpliedVol'].min()
                    })
            
            metrics['skew_analysis'] = skew_metrics
            
            # Overall surface statistics
            metrics['surface_stats'] = {
                'min_vol': surface_data['ImpliedVol'].min(),
                'max_vol': surface_data['ImpliedVol'].max(),
                'mean_vol': surface_data['ImpliedVol'].mean(),
                'vol_std': surface_data['ImpliedVol'].std(),
                'expiry_range': [surface_data['Expiry'].min(), surface_data['Expiry'].max()],
                'strike_range': [surface_data['Strike'].min(), surface_data['Strike'].max()],
                'data_points': len(surface_data)
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating surface metrics: {str(e)}")
            return {}
    
    def create_surface_metrics_dashboard(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create dashboard view of surface metrics"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ATM Term Structure', 'Skew by Expiry', 
                          'Convexity Analysis', 'Volatility Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # ATM Term Structure
        if 'atm_term_structure' in metrics:
            atm_data = metrics['atm_term_structure']
            expiries = [d['expiry'] for d in atm_data]
            atm_vols = [d['atm_vol'] for d in atm_data]
            
            fig.add_trace(
                go.Scatter(
                    x=expiries,
                    y=atm_vols,
                    mode='lines+markers',
                    name='ATM Vol',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Skew Analysis
        if 'skew_analysis' in metrics:
            skew_data = metrics['skew_analysis']
            expiries = [d['expiry'] for d in skew_data]
            skews = [d['skew'] for d in skew_data]
            
            fig.add_trace(
                go.Bar(
                    x=expiries,
                    y=skews,
                    name='Skew',
                    marker_color='red'
                ),
                row=1, col=2
            )
        
        # Convexity Analysis
        if 'skew_analysis' in metrics:
            convexities = [d['convexity'] for d in skew_data]
            
            fig.add_trace(
                go.Scatter(
                    x=expiries,
                    y=convexities,
                    mode='lines+markers',
                    name='Convexity',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # Statistics Table
        if 'surface_stats' in metrics:
            stats = metrics['surface_stats']
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value']),
                    cells=dict(values=[
                        ['Min Vol', 'Max Vol', 'Mean Vol', 'Vol Std', 'Data Points'],
                        [f"{stats['min_vol']:.3f}", f"{stats['max_vol']:.3f}", 
                         f"{stats['mean_vol']:.3f}", f"{stats['vol_std']:.3f}",
                         f"{stats['data_points']}"]
                    ])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Volatility Surface Metrics Dashboard",
            template="plotly_dark",
            height=700,
            showlegend=False
        )
        
        return fig
    
    def create_vol_surface_animation(self, historical_surfaces: Dict[str, pd.DataFrame],
                                   title: str = "Volatility Surface Evolution") -> go.Figure:
        """Create animated volatility surface over time"""
        
        dates = sorted(historical_surfaces.keys())
        
        # Create frames for animation
        frames = []
        
        for date in dates:
            surface_data = historical_surfaces[date]
            
            # Prepare data for surface
            strikes = sorted(surface_data['Strike'].unique())
            expiries = sorted(surface_data['Expiry'].unique())
            
            strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
            vol_grid = np.zeros_like(strike_grid)
            
            for i, expiry in enumerate(expiries):
                for j, strike in enumerate(strikes):
                    subset = surface_data[
                        (abs(surface_data['Expiry'] - expiry) < 0.01) & 
                        (abs(surface_data['Strike'] - strike) < 0.01)
                    ]
                    
                    if not subset.empty:
                        vol_grid[i, j] = subset['ImpliedVol'].iloc[0]
                    else:
                        vol_grid[i, j] = surface_data['ImpliedVol'].mean()
            
            frame = go.Frame(
                data=[go.Surface(
                    z=vol_grid,
                    x=strike_grid,
                    y=expiry_grid,
                    colorscale=self.color_scales['volatility']
                )],
                name=date
            )
            frames.append(frame)
        
        # Create initial surface
        initial_data = historical_surfaces[dates[0]]
        strikes = sorted(initial_data['Strike'].unique())
        expiries = sorted(initial_data['Expiry'].unique())
        
        strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
        vol_grid = np.zeros_like(strike_grid)
        
        for i, expiry in enumerate(expiries):
            for j, strike in enumerate(strikes):
                subset = initial_data[
                    (abs(initial_data['Expiry'] - expiry) < 0.01) & 
                    (abs(initial_data['Strike'] - strike) < 0.01)
                ]
                
                if not subset.empty:
                    vol_grid[i, j] = subset['ImpliedVol'].iloc[0]
                else:
                    vol_grid[i, j] = initial_data['ImpliedVol'].mean()
        
        fig = go.Figure(
            data=[go.Surface(
                z=vol_grid,
                x=strike_grid,
                y=expiry_grid,
                colorscale=self.color_scales['volatility']
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Time to Expiry',
                zaxis_title='Implied Volatility'
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 1000, "redraw": True},
                                     "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}])
                ]
            )],
            template="plotly_dark"
        )
        
        return fig
