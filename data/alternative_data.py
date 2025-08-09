import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Any
import streamlit as st
from config.settings import SATELLITE_API_KEY, SATELLITE_PROVIDERS
import trafilatura

class SatelliteDataProvider:
    """Satellite imagery and geospatial data provider"""
    
    def __init__(self):
        self.api_key = SATELLITE_API_KEY
        self.providers = SATELLITE_PROVIDERS
    
    def get_retail_footfall_data(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get synthetic retail footfall data (simulated satellite analysis)"""
        try:
            # Generate synthetic footfall data based on realistic patterns
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Base footfall with seasonal and weekly patterns
            base_footfall = 1000
            data = []
            
            for date in date_range:
                # Weekly pattern (higher on weekends)
                week_multiplier = 1.3 if date.weekday() >= 5 else 1.0
                
                # Seasonal pattern (higher during holidays)
                month_multiplier = 1.2 if date.month in [11, 12] else 1.0
                
                # Random variation
                random_multiplier = np.random.uniform(0.8, 1.2)
                
                footfall = base_footfall * week_multiplier * month_multiplier * random_multiplier
                
                data.append({
                    'date': date,
                    'location': location,
                    'footfall_count': int(footfall),
                    'parking_occupancy': np.random.uniform(0.3, 0.9),
                    'traffic_density': np.random.uniform(0.4, 0.8),
                    'economic_activity_index': footfall / base_footfall
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            # Add moving averages for trend analysis
            df['footfall_ma_7'] = df['footfall_count'].rolling(window=7).mean()
            df['footfall_ma_30'] = df['footfall_count'].rolling(window=30).mean()
            
            return df
            
        except Exception as e:
            st.error(f"Error generating retail footfall data: {str(e)}")
            return pd.DataFrame()
    
    def get_construction_activity(self, region: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get construction activity data from satellite imagery analysis"""
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='W')
            
            data = []
            base_activity = 100  # Base construction activity index
            
            for date in date_range:
                # Simulate construction cycles and seasonal effects
                activity_trend = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)  # Annual cycle
                weather_impact = 0.9 if date.month in [12, 1, 2] else 1.0  # Winter slowdown
                random_variation = np.random.uniform(0.8, 1.2)
                
                activity_index = base_activity * activity_trend * weather_impact * random_variation
                
                data.append({
                    'date': date,
                    'region': region,
                    'construction_activity_index': activity_index,
                    'new_construction_sites': np.random.poisson(5),
                    'completed_projects': np.random.poisson(2),
                    'active_crane_count': np.random.poisson(15),
                    'earth_movement_volume': np.random.uniform(1000, 5000)  # cubic meters
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error generating construction activity data: {str(e)}")
            return pd.DataFrame()
    
    def get_agricultural_yields(self, crop_type: str, region: str, year: int) -> pd.DataFrame:
        """Get agricultural yield estimates from satellite imagery"""
        try:
            # Generate monthly data for growing season
            months = list(range(3, 11))  # March to October
            data = []
            
            base_yield = {'corn': 180, 'wheat': 50, 'soybeans': 50}.get(crop_type, 100)
            
            for month in months:
                # Simulate growing season progression
                growth_stage = (month - 3) / 7  # 0 to 1 over growing season
                
                # Weather impact (random but realistic)
                weather_factor = np.random.uniform(0.8, 1.2)
                
                # NDVI (Normalized Difference Vegetation Index) simulation
                ndvi = 0.3 + 0.5 * growth_stage * weather_factor
                ndvi = min(ndvi, 0.9)  # Cap at maximum vegetation vigor
                
                # Yield prediction based on NDVI and growth stage
                predicted_yield = base_yield * ndvi * weather_factor
                
                data.append({
                    'month': month,
                    'year': year,
                    'crop_type': crop_type,
                    'region': region,
                    'ndvi': ndvi,
                    'predicted_yield_per_hectare': predicted_yield,
                    'precipitation_mm': np.random.uniform(20, 150),
                    'temperature_avg': np.random.uniform(15, 30),
                    'drought_stress_index': max(0, np.random.uniform(-0.2, 0.5))
                })
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            st.error(f"Error generating agricultural yield data: {str(e)}")
            return pd.DataFrame()
    
    def get_oil_storage_levels(self, facility: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get oil storage tank levels from satellite imagery analysis"""
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            data = []
            base_capacity = 1000000  # barrels
            current_level = 0.7  # Start at 70% capacity
            
            for date in date_range:
                # Simulate oil storage level changes
                # Weekly pattern (lower on weekends due to reduced operations)
                weekly_factor = 0.95 if date.weekday() >= 5 else 1.0
                
                # Random daily variation
                daily_change = np.random.uniform(-0.05, 0.05) * weekly_factor
                current_level = max(0.1, min(0.95, current_level + daily_change))
                
                # Convert to actual measurements
                volume_barrels = current_level * base_capacity
                
                data.append({
                    'date': date,
                    'facility': facility,
                    'storage_level_pct': current_level * 100,
                    'volume_barrels': volume_barrels,
                    'tank_shadows_detected': np.random.choice([True, False], p=[0.8, 0.2]),
                    'floating_roof_height': np.random.uniform(2, 8),  # meters
                    'estimated_ullage': (1 - current_level) * base_capacity
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            # Add moving averages
            df['level_ma_7'] = df['storage_level_pct'].rolling(window=7).mean()
            df['level_ma_30'] = df['storage_level_pct'].rolling(window=30).mean()
            
            return df
            
        except Exception as e:
            st.error(f"Error generating oil storage data: {str(e)}")
            return pd.DataFrame()

class ESGDataProvider:
    """Environmental, Social, and Governance data provider"""
    
    def get_carbon_emissions_data(self, company: str, year: int) -> Dict[str, Any]:
        """Get carbon emissions and environmental data"""
        try:
            # Simulate carbon emissions data
            base_emissions = np.random.uniform(50000, 500000)  # tons CO2 equivalent
            
            emissions_data = {
                'company': company,
                'year': year,
                'scope1_emissions': base_emissions * 0.4,  # Direct emissions
                'scope2_emissions': base_emissions * 0.3,  # Indirect emissions from energy
                'scope3_emissions': base_emissions * 0.3,  # Other indirect emissions
                'total_emissions': base_emissions,
                'emissions_intensity': base_emissions / np.random.uniform(1000, 10000),  # per unit revenue
                'renewable_energy_pct': np.random.uniform(10, 80),
                'water_usage_cubic_meters': np.random.uniform(100000, 1000000),
                'waste_generated_tons': np.random.uniform(1000, 50000),
                'waste_recycled_pct': np.random.uniform(20, 90),
                'esg_score': np.random.uniform(30, 90),
                'carbon_neutral_target_year': np.random.choice([2030, 2040, 2050, None]),
                'green_revenue_pct': np.random.uniform(5, 50)
            }
            
            return emissions_data
            
        except Exception as e:
            st.error(f"Error generating ESG data: {str(e)}")
            return {}
    
    def get_supply_chain_risk_data(self, company: str) -> pd.DataFrame:
        """Get supply chain risk assessment data"""
        try:
            # Generate supply chain risk data
            risk_factors = [
                'Geopolitical Risk', 'Natural Disasters', 'Labor Disputes',
                'Regulatory Changes', 'Currency Fluctuation', 'Cyber Security',
                'Climate Change', 'Supplier Concentration', 'Transportation Risks'
            ]
            
            regions = ['North America', 'Europe', 'Asia-Pacific', 'Latin America', 'Africa']
            
            data = []
            for region in regions:
                for risk_factor in risk_factors:
                    risk_score = np.random.uniform(1, 10)  # 1 = low risk, 10 = high risk
                    
                    data.append({
                        'company': company,
                        'region': region,
                        'risk_factor': risk_factor,
                        'risk_score': risk_score,
                        'impact_potential': np.random.choice(['Low', 'Medium', 'High']),
                        'probability': np.random.uniform(0.1, 0.8),
                        'mitigation_measures': np.random.choice(['None', 'Basic', 'Comprehensive']),
                        'supplier_count': np.random.randint(5, 100),
                        'revenue_exposure_pct': np.random.uniform(1, 25)
                    })
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            st.error(f"Error generating supply chain risk data: {str(e)}")
            return pd.DataFrame()

class EconomicIndicatorsProvider:
    """Economic indicators and alternative data provider"""
    
    def get_job_postings_data(self, company: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get job postings data as economic indicator"""
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='W')
            
            data = []
            base_postings = np.random.randint(50, 500)
            
            for date in date_range:
                # Simulate seasonal hiring patterns
                seasonal_factor = 1.2 if date.month in [1, 9] else 1.0  # January and September hiring
                economic_cycle = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
                random_variation = np.random.uniform(0.8, 1.2)
                
                weekly_postings = int(base_postings * seasonal_factor * economic_cycle * random_variation)
                
                data.append({
                    'date': date,
                    'company': company,
                    'new_job_postings': weekly_postings,
                    'tech_roles_pct': np.random.uniform(20, 60),
                    'remote_roles_pct': np.random.uniform(10, 80),
                    'senior_roles_pct': np.random.uniform(15, 40),
                    'avg_salary_posted': np.random.uniform(60000, 150000),
                    'job_posting_urgency_score': np.random.uniform(1, 10),
                    'skills_demand_index': np.random.uniform(50, 150)
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            # Add moving averages
            df['postings_ma_4w'] = df['new_job_postings'].rolling(window=4).mean()
            df['postings_ma_12w'] = df['new_job_postings'].rolling(window=12).mean()
            
            return df
            
        except Exception as e:
            st.error(f"Error generating job postings data: {str(e)}")
            return pd.DataFrame()
    
    def get_patent_filings_data(self, company: str, technology_area: str, year: int) -> pd.DataFrame:
        """Get patent filings data as innovation indicator"""
        try:
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
            
            data = []
            base_patents = np.random.randint(10, 100)
            
            for quarter in quarters:
                quarterly_filings = int(base_patents * np.random.uniform(0.7, 1.3))
                
                data.append({
                    'year': year,
                    'quarter': quarter,
                    'company': company,
                    'technology_area': technology_area,
                    'patent_applications': quarterly_filings,
                    'patents_granted': int(quarterly_filings * np.random.uniform(0.3, 0.7)),
                    'international_filings': int(quarterly_filings * np.random.uniform(0.2, 0.6)),
                    'ai_ml_related': int(quarterly_filings * np.random.uniform(0.1, 0.4)),
                    'collaboration_patents': int(quarterly_filings * np.random.uniform(0.05, 0.2)),
                    'patent_citation_score': np.random.uniform(1, 10),
                    'innovation_index': quarterly_filings * np.random.uniform(0.8, 1.2)
                })
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            st.error(f"Error generating patent data: {str(e)}")
            return pd.DataFrame()
    
    def get_corporate_flight_data(self, company: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get corporate flight activity as business activity indicator"""
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            data = []
            for date in date_range:
                # Business days have more activity
                business_factor = 1.5 if date.weekday() < 5 else 0.3
                
                # Monthly business cycle
                monthly_factor = 1.2 if date.day <= 15 else 0.8
                
                # Random daily flights
                daily_flights = int(np.random.poisson(3) * business_factor * monthly_factor)
                
                if daily_flights > 0:
                    data.append({
                        'date': date,
                        'company': company,
                        'flight_count': daily_flights,
                        'domestic_flights': int(daily_flights * np.random.uniform(0.6, 0.9)),
                        'international_flights': daily_flights - int(daily_flights * np.random.uniform(0.6, 0.9)),
                        'avg_flight_duration': np.random.uniform(1, 8),  # hours
                        'executive_travel_index': np.random.uniform(1, 10),
                        'business_activity_score': daily_flights * np.random.uniform(0.8, 1.2)
                    })
            
            if data:
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)
                
                # Add moving averages
                df['flights_ma_7'] = df['flight_count'].rolling(window=7).mean()
                df['flights_ma_30'] = df['flight_count'].rolling(window=30).mean()
                
                return df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error generating corporate flight data: {str(e)}")
            return pd.DataFrame()

class WebScrapingProvider:
    """Web scraping for financial news and alternative data"""
    
    def scrape_financial_news(self, url: str) -> Dict[str, Any]:
        """Scrape financial news content"""
        try:
            downloaded = trafilatura.fetch_url(url)
            text_content = trafilatura.extract(downloaded)
            
            if text_content:
                return {
                    'url': url,
                    'content': text_content,
                    'scraped_at': datetime.now(),
                    'word_count': len(text_content.split()),
                    'content_length': len(text_content)
                }
            else:
                return {'error': 'Failed to extract content'}
                
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return {'error': str(e)}
    
    def get_sec_filings_data(self, company_ticker: str) -> pd.DataFrame:
        """Get SEC filings data (simulated)"""
        try:
            # Simulate SEC filings data
            filing_types = ['10-K', '10-Q', '8-K', 'DEF 14A', 'S-1']
            
            data = []
            base_date = datetime.now() - timedelta(days=365)
            
            for i in range(20):  # Last 20 filings
                filing_date = base_date + timedelta(days=i*18)
                filing_type = np.random.choice(filing_types)
                
                data.append({
                    'filing_date': filing_date,
                    'company_ticker': company_ticker,
                    'filing_type': filing_type,
                    'document_size_mb': np.random.uniform(0.5, 50),
                    'pages_count': np.random.randint(10, 500),
                    'amendments_count': np.random.randint(0, 3),
                    'key_metrics_updated': np.random.choice([True, False], p=[0.7, 0.3]),
                    'material_events': np.random.choice([True, False], p=[0.3, 0.7]),
                    'insider_trading_disclosed': np.random.choice([True, False], p=[0.2, 0.8])
                })
            
            df = pd.DataFrame(data)
            df.set_index('filing_date', inplace=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error generating SEC filings data: {str(e)}")
            return pd.DataFrame()

class AlternativeDataAggregator:
    """Aggregate and analyze alternative data sources"""
    
    def __init__(self):
        self.satellite_provider = SatelliteDataProvider()
        self.esg_provider = ESGDataProvider()
        self.economic_provider = EconomicIndicatorsProvider()
        self.web_provider = WebScrapingProvider()
    
    def create_company_alternative_score(self, company: str, ticker: str) -> Dict[str, Any]:
        """Create comprehensive alternative data score for a company"""
        try:
            # Get various alternative data points
            current_year = datetime.now().year
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # ESG Score
            esg_data = self.esg_provider.get_carbon_emissions_data(company, current_year)
            esg_score = esg_data.get('esg_score', 50)
            
            # Job Postings Growth
            job_data = self.economic_provider.get_job_postings_data(company, start_date, end_date)
            if not job_data.empty:
                job_growth = (job_data['new_job_postings'].tail(4).mean() / 
                            job_data['new_job_postings'].head(4).mean() - 1) * 100
            else:
                job_growth = 0
            
            # Patent Innovation Score
            patent_data = self.economic_provider.get_patent_filings_data(company, 'Technology', current_year)
            if not patent_data.empty:
                innovation_score = patent_data['innovation_index'].mean()
            else:
                innovation_score = 50
            
            # Corporate Activity Score
            flight_data = self.economic_provider.get_corporate_flight_data(company, start_date, end_date)
            if not flight_data.empty:
                activity_score = flight_data['business_activity_score'].mean()
            else:
                activity_score = 50
            
            # Supply Chain Risk
            supply_risk = self.esg_provider.get_supply_chain_risk_data(company)
            if not supply_risk.empty:
                avg_risk = supply_risk['risk_score'].mean()
                risk_score = 100 - (avg_risk * 10)  # Invert scale so higher is better
            else:
                risk_score = 50
            
            # Aggregate Score (weighted average)
            weights = {
                'esg': 0.25,
                'job_growth': 0.20,
                'innovation': 0.20,
                'activity': 0.15,
                'risk': 0.20
            }
            
            composite_score = (
                esg_score * weights['esg'] +
                max(0, min(100, 50 + job_growth)) * weights['job_growth'] +
                innovation_score * weights['innovation'] +
                activity_score * weights['activity'] +
                risk_score * weights['risk']
            )
            
            return {
                'company': company,
                'ticker': ticker,
                'composite_score': composite_score,
                'esg_score': esg_score,
                'job_growth_pct': job_growth,
                'innovation_score': innovation_score,
                'business_activity_score': activity_score,
                'supply_chain_risk_score': risk_score,
                'last_updated': datetime.now(),
                'data_quality_score': np.random.uniform(70, 95),  # Simulated data quality
                'recommendation': 'BUY' if composite_score > 70 else 'HOLD' if composite_score > 40 else 'SELL'
            }
            
        except Exception as e:
            st.error(f"Error creating alternative data score: {str(e)}")
            return {}
    
    def get_sector_alternative_trends(self, sector: str) -> pd.DataFrame:
        """Get alternative data trends for a sector"""
        try:
            # Generate sector-level alternative data trends
            date_range = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
            
            data = []
            for date in date_range:
                # Simulate sector trends
                base_trend = 100 + 10 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                data.append({
                    'date': date,
                    'sector': sector,
                    'satellite_activity_index': base_trend * np.random.uniform(0.9, 1.1),
                    'job_postings_index': base_trend * np.random.uniform(0.8, 1.2),
                    'patent_activity_index': base_trend * np.random.uniform(0.85, 1.15),
                    'esg_sentiment_index': base_trend * np.random.uniform(0.9, 1.1),
                    'supply_chain_stability_index': base_trend * np.random.uniform(0.95, 1.05),
                    'overall_alternative_score': base_trend * np.random.uniform(0.9, 1.1)
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error generating sector trends: {str(e)}")
            return pd.DataFrame()
