import os

# API Keys and Configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
SATELLITE_API_KEY = os.getenv("SATELLITE_API_KEY", "")

# Financial Parameters
RISK_FREE_RATE = 0.05  # 5% default risk-free rate
TRADING_DAYS_PER_YEAR = 252

# Model Parameters
MONTE_CARLO_SIMULATIONS = 10000
VOLATILITY_LOOKBACK_DAYS = 30

# Data Sources
FINANCIAL_NEWS_URLS = [
    "https://finance.yahoo.com/news/",
    "https://www.marketwatch.com/",
    "https://www.bloomberg.com/markets"
]

# Alternative Data Sources
SATELLITE_PROVIDERS = {
    "planet": "https://api.planet.com/",
    "orbital_insight": "https://orbitalinsight.com/api/",
    "spaceknow": "https://spaceknow.com/api/"
}

# Social Media Sources
SOCIAL_PLATFORMS = {
    "twitter": "https://api.twitter.com/2/",
    "reddit": "https://reddit.com/r/",
    "stocktwits": "https://api.stocktwits.com/api/2/"
}
