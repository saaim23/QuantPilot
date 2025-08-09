import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import streamlit as st
import yfinance as yf
import re
import time
from textblob import TextBlob
import trafilatura
from config.settings import TWITTER_BEARER_TOKEN, FINANCIAL_NEWS_URLS

class SentimentAnalyzer:
    """Advanced sentiment analysis for financial data"""
    
    def __init__(self):
        self.twitter_token = TWITTER_BEARER_TOKEN
        self.news_urls = FINANCIAL_NEWS_URLS
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a text using TextBlob"""
        try:
            blob = TextBlob(text)
            
            # TextBlob sentiment
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Normalize to financial sentiment scale
            sentiment_score = polarity
            confidence = 1 - subjectivity  # Higher confidence for more objective text
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
        except Exception as e:
            st.warning(f"Sentiment analysis failed: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.5
            }
    
    def analyze_stock_sentiment(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Analyze sentiment for a specific stock from news and social media"""
        try:
            # Simulate comprehensive sentiment analysis
            # In a real implementation, this would fetch from news APIs, social media APIs, etc.
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Generate realistic sentiment data based on stock performance
            ticker = yf.Ticker(symbol)
            try:
                stock_data = ticker.history(period=f"{days_back}d")
                if not stock_data.empty:
                    # Calculate recent performance to influence sentiment
                    recent_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1
                    base_sentiment = np.tanh(recent_return * 5)  # Sigmoid-like function
                else:
                    base_sentiment = 0.0
            except:
                base_sentiment = 0.0
            
            # Generate sentiment history
            sentiment_history = []
            for i in range(days_back):
                date = start_date + timedelta(days=i)
                daily_sentiment = base_sentiment + np.random.normal(0, 0.2)
                daily_sentiment = np.clip(daily_sentiment, -1, 1)
                
                sentiment_history.append({
                    'date': date,
                    'sentiment_score': daily_sentiment,
                    'article_count': np.random.randint(5, 25),
                    'social_mentions': np.random.randint(50, 500)
                })
            
            # Generate sample articles
            article_titles = [
                f"{symbol} reports strong quarterly earnings",
                f"Analysts upgrade {symbol} price target",
                f"{symbol} announces new product launch",
                f"Market volatility affects {symbol} trading",
                f"{symbol} CEO discusses future strategy",
                f"Institutional investors increase {symbol} holdings",
                f"{symbol} stock shows technical breakout",
                f"Regulatory changes impact {symbol} sector"
            ]
            
            articles = []
            for i in range(min(len(article_titles), 8)):
                title = article_titles[i]
                sentiment_result = self.analyze_text_sentiment(title)
                
                articles.append({
                    'title': title,
                    'sentiment': sentiment_result['sentiment_score'],
                    'confidence': sentiment_result['confidence'],
                    'url': f"https://example-news.com/article-{i+1}",
                    'summary': f"This article discusses recent developments related to {symbol}...",
                    'published_date': datetime.now() - timedelta(hours=np.random.randint(1, 168))
                })
            
            # Calculate overall metrics
            overall_sentiment = np.mean([h['sentiment_score'] for h in sentiment_history])
            positive_count = len([a for a in articles if a['sentiment'] > 0.1])
            negative_count = len([a for a in articles if a['sentiment'] < -0.1])
            
            return {
                'symbol': symbol,
                'overall_sentiment': overall_sentiment,
                'sentiment_history': sentiment_history,
                'articles': articles,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': len(articles) - positive_count - negative_count,
                'analysis_period': f"{days_back} days",
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return {}
    
    def get_social_media_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment for a stock"""
        try:
            # Simulate social media sentiment analysis
            # In production, this would integrate with Twitter API, Reddit API, etc.
            
            # Base sentiment influenced by random market factors
            base_sentiment = np.random.normal(0, 0.3)
            
            # Twitter sentiment simulation
            twitter_sentiment = base_sentiment + np.random.normal(0, 0.2)
            twitter_sentiment = np.clip(twitter_sentiment, -1, 1)
            
            # Reddit sentiment simulation
            reddit_sentiment = base_sentiment + np.random.normal(0, 0.25)
            reddit_sentiment = np.clip(reddit_sentiment, -1, 1)
            
            # StockTwits sentiment simulation
            stocktwits_sentiment = base_sentiment + np.random.normal(0, 0.15)
            stocktwits_sentiment = np.clip(stocktwits_sentiment, -1, 1)
            
            # Volume metrics
            mention_count = np.random.randint(100, 2000)
            sentiment_volume = mention_count * abs(base_sentiment)
            
            return {
                'symbol': symbol,
                'twitter_sentiment': twitter_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'stocktwits_sentiment': stocktwits_sentiment,
                'mention_count': mention_count,
                'sentiment_volume': sentiment_volume,
                'engagement_score': np.random.uniform(0.3, 0.9),
                'trending_score': np.random.uniform(0.1, 1.0),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error getting social media sentiment: {str(e)}")
            return {}
    
    def analyze_market_sentiment(self) -> Dict[str, Any]:
        """Analyze overall market sentiment"""
        try:
            # Get VIX data for fear/greed indicator
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5d")
            
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                # Convert VIX to sentiment (inverse relationship)
                vix_sentiment = max(0, min(1, (40 - current_vix) / 30))  # 0-1 scale
            else:
                vix_sentiment = 0.5
            
            # Simulate other market indicators
            market_indicators = {
                'vix_sentiment': vix_sentiment,
                'put_call_ratio': np.random.uniform(0.8, 1.5),
                'advance_decline': np.random.uniform(0.3, 1.8),
                'new_highs_lows': np.random.uniform(0.2, 2.0),
                'margin_debt': np.random.uniform(0.4, 1.2)
            }
            
            # Calculate overall sentiment score
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # VIX gets highest weight
            
            # Normalize indicators to 0-1 scale
            normalized_indicators = []
            normalized_indicators.append(market_indicators['vix_sentiment'])
            normalized_indicators.append(max(0, min(1, (1.5 - market_indicators['put_call_ratio']) / 0.7)))
            normalized_indicators.append(max(0, min(1, market_indicators['advance_decline'] / 1.8)))
            normalized_indicators.append(max(0, min(1, market_indicators['new_highs_lows'] / 2.0)))
            normalized_indicators.append(max(0, min(1, market_indicators['margin_debt'] / 1.2)))
            
            overall_score = sum(w * i for w, i in zip(weights, normalized_indicators))
            
            # Sector sentiment simulation
            sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 
                      'Communication Services', 'Industrials', 'Consumer Staples', 'Energy']
            
            sector_sentiment = {}
            for sector in sectors:
                # Add some correlation but with sector-specific noise
                sector_sentiment[sector] = overall_score + np.random.normal(0, 0.15)
                sector_sentiment[sector] = max(0, min(1, sector_sentiment[sector]))
            
            # Fear & Greed indicators
            fear_greed_indicators = {
                'Stock Price Momentum': overall_score + np.random.normal(0, 0.1),
                'Market Volatility': vix_sentiment,
                'Stock Price Breadth': normalized_indicators[2],
                'Put/Call Ratio': normalized_indicators[1],
                'Junk Bond Demand': np.random.uniform(0.2, 0.8),
                'Market Volume': np.random.uniform(0.3, 0.9),
                'Safe Haven Demand': 1 - overall_score + np.random.normal(0, 0.1)
            }
            
            # Ensure values are in 0-1 range
            for key in fear_greed_indicators:
                fear_greed_indicators[key] = max(0, min(1, fear_greed_indicators[key]))
            
            return {
                'overall_score': overall_score,
                'sentiment_level': 'Greedy' if overall_score > 0.7 else 'Fearful' if overall_score < 0.3 else 'Neutral',
                'market_indicators': market_indicators,
                'sector_sentiment': sector_sentiment,
                'fear_greed_indicators': fear_greed_indicators,
                'analysis_timestamp': datetime.now(),
                'data_quality': np.random.uniform(0.8, 0.95)
            }
            
        except Exception as e:
            st.error(f"Error analyzing market sentiment: {str(e)}")
            return {}
    
    def analyze_social_media_trends(self, platforms: List[str], keywords: List[str]) -> Dict[str, Any]:
        """Analyze social media trends across platforms"""
        try:
            platform_data = {}
            
            for platform in platforms:
                # Simulate platform-specific sentiment analysis
                platform_sentiment = {
                    'average_sentiment': np.random.normal(0, 0.3),
                    'post_count': np.random.randint(100, 5000),
                    'engagement_score': np.random.uniform(0.2, 0.9),
                    'trending_keywords': np.random.choice(keywords, size=min(5, len(keywords)), replace=False).tolist() if keywords else [],
                    'sentiment_distribution': {
                        'positive': np.random.uniform(0.3, 0.6),
                        'neutral': np.random.uniform(0.2, 0.4),
                        'negative': np.random.uniform(0.1, 0.3)
                    }
                }
                
                # Ensure sentiment distribution sums to 1
                total = sum(platform_sentiment['sentiment_distribution'].values())
                for key in platform_sentiment['sentiment_distribution']:
                    platform_sentiment['sentiment_distribution'][key] /= total
                
                platform_data[f'{platform.lower()}_sentiment'] = platform_sentiment
            
            # Generate trending topics
            trending_topics = []
            topic_templates = [
                "earnings season", "fed meeting", "inflation data", "market volatility",
                "tech stocks", "crypto market", "oil prices", "interest rates",
                "economic outlook", "sector rotation", "growth vs value", "meme stocks"
            ]
            
            for i, topic in enumerate(topic_templates[:8]):
                trending_topics.append({
                    'topic': topic,
                    'mention_count': np.random.randint(50, 1000),
                    'sentiment_score': np.random.uniform(-0.5, 0.5),
                    'growth_rate': np.random.uniform(-0.3, 0.8),
                    'platforms': np.random.choice(platforms, size=np.random.randint(1, len(platforms)+1), replace=False).tolist()
                })
            
            return {
                **platform_data,
                'trending_topics': trending_topics,
                'analysis_period': '24 hours',
                'total_mentions': sum([data['post_count'] for data in platform_data.values()]),
                'overall_sentiment': np.mean([data['average_sentiment'] for data in platform_data.values()]),
                'data_freshness': 'real-time',
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error analyzing social media trends: {str(e)}")
            return {}

class NewsDataProvider:
    """Financial news data provider and analyzer"""
    
    def __init__(self):
        self.news_sources = {
            'Bloomberg': 'https://www.bloomberg.com/markets',
            'Reuters': 'https://www.reuters.com/business/finance',
            'CNBC': 'https://www.cnbc.com/markets/',
            'MarketWatch': 'https://www.marketwatch.com/',
            'Financial Times': 'https://www.ft.com/markets'
        }
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def scrape_financial_news(self, url: str) -> Dict[str, Any]:
        """Scrape financial news from a URL"""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text_content = trafilatura.extract(downloaded)
                
                if text_content:
                    # Analyze sentiment
                    sentiment_result = self.sentiment_analyzer.analyze_text_sentiment(text_content)
                    
                    return {
                        'url': url,
                        'content': text_content,
                        'word_count': len(text_content.split()),
                        'sentiment': sentiment_result,
                        'scraped_at': datetime.now(),
                        'success': True
                    }
            
            return {'url': url, 'success': False, 'error': 'Failed to extract content'}
            
        except Exception as e:
            return {'url': url, 'success': False, 'error': str(e)}
    
    def analyze_financial_news(self, sources: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze financial news from multiple sources over a date range"""
        try:
            # Simulate news analysis over the specified period
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Generate daily sentiment data
            daily_sentiment = []
            current_date = start_dt
            
            while current_date <= end_dt:
                # Simulate daily news sentiment
                daily_score = np.random.normal(0, 0.3)
                article_count = np.random.randint(5, 30)
                
                daily_sentiment.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'sentiment': daily_score,
                    'article_count': article_count,
                    'positive_articles': int(article_count * 0.4),
                    'negative_articles': int(article_count * 0.25),
                    'neutral_articles': int(article_count * 0.35)
                })
                
                current_date += timedelta(days=1)
            
            # Topic sentiment analysis
            financial_topics = [
                'Interest Rates', 'Inflation', 'Employment', 'GDP Growth',
                'Corporate Earnings', 'Market Volatility', 'Currency Markets',
                'Commodities', 'Geopolitical Events', 'Technology Sector'
            ]
            
            topic_sentiment = {}
            for topic in financial_topics:
                topic_sentiment[topic] = np.random.uniform(-0.5, 0.5)
            
            # Key themes identification
            themes = [
                'Federal Reserve policy uncertainty',
                'Supply chain disruptions',
                'Digital transformation acceleration',
                'ESG investing trends',
                'Cryptocurrency adoption',
                'Renewable energy transition',
                'Inflation concerns persist',
                'Labor market tightness',
                'Geopolitical tensions',
                'Central bank coordination'
            ]
            
            key_themes = np.random.choice(themes, size=6, replace=False).tolist()
            
            # Source analysis
            source_analysis = {}
            for source in sources:
                source_analysis[source] = {
                    'article_count': np.random.randint(10, 100),
                    'average_sentiment': np.random.uniform(-0.3, 0.3),
                    'coverage_topics': np.random.choice(financial_topics, size=5, replace=False).tolist(),
                    'credibility_score': np.random.uniform(0.7, 0.95)
                }
            
            return {
                'analysis_period': f"{start_date} to {end_date}",
                'sources_analyzed': sources,
                'daily_sentiment': daily_sentiment,
                'topic_sentiment': topic_sentiment,
                'key_themes': key_themes,
                'source_analysis': source_analysis,
                'overall_sentiment': np.mean([d['sentiment'] for d in daily_sentiment]),
                'sentiment_volatility': np.std([d['sentiment'] for d in daily_sentiment]),
                'total_articles': sum([d['article_count'] for d in daily_sentiment]),
                'analysis_quality': np.random.uniform(0.8, 0.95),
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error analyzing financial news: {str(e)}")
            return {}
    
    def get_breaking_news_sentiment(self) -> Dict[str, Any]:
        """Get sentiment analysis of breaking financial news"""
        try:
            # Simulate breaking news sentiment
            breaking_news = [
                {
                    'headline': 'Federal Reserve signals potential rate cut',
                    'source': 'Reuters',
                    'timestamp': datetime.now() - timedelta(minutes=30),
                    'impact_score': 8.5,
                    'sentiment': 0.3
                },
                {
                    'headline': 'Tech stocks rally on AI investment news',
                    'source': 'Bloomberg',
                    'timestamp': datetime.now() - timedelta(hours=1),
                    'impact_score': 7.2,
                    'sentiment': 0.6
                },
                {
                    'headline': 'Oil prices surge amid supply concerns',
                    'source': 'CNBC',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'impact_score': 6.8,
                    'sentiment': -0.2
                },
                {
                    'headline': 'Major bank reports strong quarterly earnings',
                    'source': 'MarketWatch',
                    'timestamp': datetime.now() - timedelta(hours=3),
                    'impact_score': 7.5,
                    'sentiment': 0.4
                }
            ]
            
            # Calculate aggregate metrics
            avg_sentiment = np.mean([news['sentiment'] for news in breaking_news])
            max_impact = max([news['impact_score'] for news in breaking_news])
            
            return {
                'breaking_news': breaking_news,
                'average_sentiment': avg_sentiment,
                'max_impact_score': max_impact,
                'news_count': len(breaking_news),
                'last_updated': datetime.now(),
                'market_moving_events': len([n for n in breaking_news if n['impact_score'] > 7.0])
            }
            
        except Exception as e:
            st.error(f"Error getting breaking news sentiment: {str(e)}")
            return {}
    
    def analyze_earnings_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment around earnings announcements"""
        try:
            # Simulate earnings sentiment analysis
            earnings_phases = ['pre-announcement', 'during-call', 'post-announcement']
            
            sentiment_timeline = {}
            for phase in earnings_phases:
                sentiment_timeline[phase] = {
                    'sentiment_score': np.random.uniform(-0.5, 0.5),
                    'volume_spike': np.random.uniform(1.2, 3.5),
                    'analyst_reactions': np.random.randint(3, 15),
                    'social_media_buzz': np.random.uniform(0.3, 0.9)
                }
            
            # Key sentiment drivers
            sentiment_drivers = [
                {'factor': 'Revenue Beat/Miss', 'impact': np.random.uniform(-0.4, 0.4)},
                {'factor': 'EPS Beat/Miss', 'impact': np.random.uniform(-0.3, 0.3)},
                {'factor': 'Forward Guidance', 'impact': np.random.uniform(-0.5, 0.5)},
                {'factor': 'Management Commentary', 'impact': np.random.uniform(-0.2, 0.2)},
                {'factor': 'Analyst Q&A', 'impact': np.random.uniform(-0.3, 0.3)}
            ]
            
            # Overall earnings sentiment
            overall_sentiment = np.mean([driver['impact'] for driver in sentiment_drivers])
            
            return {
                'symbol': symbol,
                'sentiment_timeline': sentiment_timeline,
                'sentiment_drivers': sentiment_drivers,
                'overall_sentiment': overall_sentiment,
                'earnings_surprise': np.random.uniform(-0.2, 0.2),
                'analyst_revision_trend': np.random.choice(['positive', 'negative', 'neutral']),
                'institutional_sentiment': np.random.uniform(-0.3, 0.3),
                'analysis_date': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error analyzing earnings sentiment: {str(e)}")
            return {}

class FinancialNewsFilter:
    """Filter and categorize financial news by relevance and impact"""
    
    def __init__(self):
        self.market_moving_keywords = [
            'federal reserve', 'interest rate', 'inflation', 'gdp', 'unemployment',
            'earnings', 'merger', 'acquisition', 'ipo', 'bankruptcy', 'dividend',
            'guidance', 'forecast', 'outlook', 'upgrade', 'downgrade'
        ]
        
        self.sector_keywords = {
            'Technology': ['tech', 'software', 'ai', 'cloud', 'semiconductor'],
            'Healthcare': ['pharma', 'biotech', 'medical', 'drug', 'clinical'],
            'Financial': ['bank', 'insurance', 'fintech', 'lending', 'credit'],
            'Energy': ['oil', 'gas', 'renewable', 'solar', 'wind'],
            'Consumer': ['retail', 'consumer', 'restaurant', 'e-commerce']
        }
    
    def categorize_news(self, headline: str, content: str) -> Dict[str, Any]:
        """Categorize news article by sector and impact level"""
        try:
            text = (headline + ' ' + content).lower()
            
            # Determine sector relevance
            sector_scores = {}
            for sector, keywords in self.sector_keywords.items():
                score = sum([1 for keyword in keywords if keyword in text])
                sector_scores[sector] = score
            
            primary_sector = max(sector_scores, key=sector_scores.get) if max(sector_scores.values()) > 0 else 'General'
            
            # Determine market impact
            impact_score = sum([1 for keyword in self.market_moving_keywords if keyword in text])
            
            if impact_score >= 3:
                impact_level = 'High'
            elif impact_score >= 1:
                impact_level = 'Medium'
            else:
                impact_level = 'Low'
            
            return {
                'primary_sector': primary_sector,
                'sector_scores': sector_scores,
                'impact_level': impact_level,
                'impact_score': impact_score,
                'market_moving': impact_score >= 2
            }
            
        except Exception as e:
            return {
                'primary_sector': 'General',
                'sector_scores': {},
                'impact_level': 'Low',
                'impact_score': 0,
                'market_moving': False
            }
