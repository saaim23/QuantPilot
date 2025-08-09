# Deployment Guide

## Overview

This guide covers multiple deployment options for the Quantitative Finance Platform, from local development to production cloud deployments.

## Local Development

### Prerequisites
- Python 3.11 or higher
- Git
- 4GB+ RAM recommended
- Internet connection for market data

### Quick Start
```bash
# Clone repository
git clone https://github.com/your-username/quantitative-finance-platform.git
cd quantitative-finance-platform

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py --server.port 5000
```

Access the application at `http://localhost:5000`

## Cloud Deployment Options

### 1. Streamlit Cloud (Recommended for Demo)

**Pros:** Free, easy setup, automatic updates from GitHub
**Cons:** Limited resources, public repositories only

**Steps:**
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub account
4. Select repository and branch
5. Click "Deploy"

**Configuration:**
Create `.streamlit/secrets.toml` for API keys:
```toml
[secrets]
ALPHA_VANTAGE_API_KEY = "your-api-key"
TWITTER_BEARER_TOKEN = "your-bearer-token"
```

### 2. Heroku Deployment

**Pros:** Easy scaling, add-ons available, free tier
**Cons:** Pricing changes, cold starts

**Setup Files:**

`Procfile`:
```
web: streamlit run app.py --server.port $PORT --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```

`runtime.txt`:
```
python-3.11.0
```

`requirements.txt`: Use the existing requirements file

**Deployment Steps:**
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set config vars
heroku config:set ALPHA_VANTAGE_API_KEY=your-key
heroku config:set TWITTER_BEARER_TOKEN=your-token

# Deploy
git push heroku main
```

### 3. AWS Deployment

#### Option A: AWS EC2

**Setup:**
```bash
# Launch EC2 instance (Ubuntu 20.04 LTS)
# Connect via SSH

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.11 python3-pip git -y

# Clone repository
git clone https://github.com/your-username/quantitative-finance-platform.git
cd quantitative-finance-platform

# Install dependencies
pip3 install -r requirements.txt

# Install nginx for reverse proxy
sudo apt install nginx -y

# Create systemd service
sudo nano /etc/systemd/system/quantfinance.service
```

`/etc/systemd/system/quantfinance.service`:
```ini
[Unit]
Description=Quantitative Finance Platform
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/quantitative-finance-platform
Environment=PATH=/home/ubuntu/.local/bin
ExecStart=/home/ubuntu/.local/bin/streamlit run app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

**Nginx Configuration:**
`/etc/nginx/sites-available/quantfinance`:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Start Services:**
```bash
# Enable and start service
sudo systemctl enable quantfinance
sudo systemctl start quantfinance

# Enable nginx site
sudo ln -s /etc/nginx/sites-available/quantfinance /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### Option B: AWS ECS with Docker

`Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

`docker-compose.yml`:
```yaml
version: '3.8'

services:
  quantfinance:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}
      - TWITTER_BEARER_TOKEN=${TWITTER_BEARER_TOKEN}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 5
```

### 4. Google Cloud Platform

#### Cloud Run Deployment

`cloudbuild.yaml`:
```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/quantfinance', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/quantfinance']
- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - 'run'
  - 'deploy'
  - 'quantfinance'
  - '--image=gcr.io/$PROJECT_ID/quantfinance'
  - '--region=us-central1'
  - '--platform=managed'
  - '--allow-unauthenticated'
```

**Deploy:**
```bash
# Set up gcloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy
gcloud builds submit --config cloudbuild.yaml
```

### 5. Digital Ocean App Platform

`app.yaml`:
```yaml
name: quantitative-finance-platform
services:
- name: web
  source_dir: /
  github:
    repo: your-username/quantitative-finance-platform
    branch: main
  run_command: streamlit run app.py --server.port $PORT --server.headless true
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: ALPHA_VANTAGE_API_KEY
    value: your-api-key
    type: SECRET
```

## Production Considerations

### Performance Optimization

**1. Resource Management:**
```python
# config/production.py
import streamlit as st

# Configure Streamlit for production
st.set_page_config(
    page_title="Quantitative Finance Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "Professional Quantitative Finance Platform"
    }
)
```

**2. Caching Configuration:**
```python
# Enhanced caching for production
@st.cache_data(ttl=300)  # 5 minutes cache
def get_market_data(symbol, period):
    return market_provider.get_stock_data(symbol, period)

@st.cache_resource
def load_ml_models():
    return MLOptionPricer()
```

**3. Error Monitoring:**
```python
# Add error tracking
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(traceback.format_exc())
            st.error(f"An error occurred: {str(e)}")
    return wrapper
```

### Security Configuration

**1. Environment Variables:**
```bash
# Production environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

**2. Secrets Management:**
```toml
# .streamlit/secrets.toml (production)
[secrets]
ALPHA_VANTAGE_API_KEY = "prod-api-key"
TWITTER_BEARER_TOKEN = "prod-bearer-token"
DATABASE_URL = "production-db-url"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
```

### Monitoring and Logging

**1. Health Check Endpoint:**
```python
# health_check.py
import streamlit as st
import requests
import time

def health_check():
    try:
        # Check market data connection
        response = requests.get("https://finance.yahoo.com", timeout=5)
        if response.status_code != 200:
            return False, "Market data source unavailable"
        
        # Check model loading
        from models.black_scholes import BlackScholesModel
        test_price = BlackScholesModel.option_price(100, 105, 0.25, 0.05, 0.20, 'call')
        if test_price <= 0:
            return False, "Model calculation failed"
        
        return True, "All systems operational"
    except Exception as e:
        return False, f"Health check failed: {str(e)}"
```

**2. Performance Monitoring:**
```python
# monitoring.py
import time
import psutil
import streamlit as st

class PerformanceMonitor:
    @staticmethod
    def log_performance(func_name, execution_time, memory_usage):
        st.sidebar.info(f"""
        **Performance Metrics**
        - Function: {func_name}
        - Execution Time: {execution_time:.2f}s
        - Memory Usage: {memory_usage:.2f} MB
        """)
    
    @staticmethod
    def monitor_function(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            PerformanceMonitor.log_performance(
                func.__name__,
                end_time - start_time,
                end_memory - start_memory
            )
            
            return result
        return wrapper
```

### Backup and Disaster Recovery

**1. Data Backup:**
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/quantfinance_$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application code
tar -czf $BACKUP_DIR/app_code.tar.gz /app

# Backup model files
cp -r /app/models/saved_models $BACKUP_DIR/

# Backup configuration
cp -r /app/.streamlit $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
```

**2. Automated Backups:**
```bash
# Add to crontab for daily backups
0 2 * * * /path/to/backup.sh
```

### SSL/TLS Configuration

**1. Let's Encrypt with Nginx:**
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

**2. Updated Nginx Configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Troubleshooting

### Common Issues

**1. Port Already in Use:**
```bash
# Find process using port
lsof -i :8501
# Kill process
kill -9 PID
```

**2. Memory Issues:**
```python
# Add memory monitoring
import gc
gc.collect()  # Force garbage collection

# Reduce cache size
@st.cache_data(max_entries=100, ttl=300)
def cached_function():
    pass
```

**3. API Rate Limits:**
```python
# Implement rate limiting
import time
from functools import wraps

def rate_limit(max_calls_per_minute=60):
    def decorator(func):
        calls = []
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [call_time for call_time in calls if now - call_time < 60]
            if len(calls) >= max_calls_per_minute:
                sleep_time = 60 - (now - calls[0])
                time.sleep(sleep_time)
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### Logs and Debugging

**Application Logs:**
```bash
# View systemd logs
sudo journalctl -u quantfinance -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Application-specific logs
tail -f /app/logs/application.log
```

**Debug Mode:**
```python
# Enable debug mode for development
import streamlit as st

if st.secrets.get("DEBUG_MODE", False):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.sidebar.write("Debug mode enabled")
```

## Scaling Considerations

### Horizontal Scaling

**1. Load Balancer Configuration:**
```nginx
upstream quantfinance_backend {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://quantfinance_backend;
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
}
```

**2. Container Orchestration:**
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantfinance-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantfinance
  template:
    metadata:
      labels:
        app: quantfinance
    spec:
      containers:
      - name: quantfinance
        image: quantfinance:latest
        ports:
        - containerPort: 8501
        env:
        - name: ALPHA_VANTAGE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: alpha-vantage-key
```

This deployment guide covers everything from simple local development to enterprise-grade production deployments. Choose the option that best fits your needs and scaling requirements.