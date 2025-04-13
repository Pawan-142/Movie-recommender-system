import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import re
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import json
import os
import torch


# Set page configuration
st.set_page_config(
    page_title="Enhanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .recommendation-buy {
        color: green;
        font-weight: bold;
        font-size: 24px;
    }
    .recommendation-sell {
        color: red;
        font-weight: bold;
        font-size: 24px;
    }
    .recommendation-hold {
        color: orange;
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'stock_data_cache' not in st.session_state:
    st.session_state.stock_data_cache = {}
if 'news_cache' not in st.session_state:
    st.session_state.news_cache = {}
if 'ml_summarizer' not in st.session_state:
    try:
        st.session_state.ml_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Error loading ML model: {str(e)}. Will use rule-based summarization instead.")
        st.session_state.ml_summarizer = None

# Function to load stock data from multiple sources
def load_stock_data():
    # Create a cache file path
    cache_file = "stock_data_cache.json"
    name_to_ticker_file = "name_to_ticker_cache.json"
    
    # Check if cache files exist and are less than 24 hours old
    if os.path.exists(cache_file) and os.path.exists(name_to_ticker_file):
        file_time = os.path.getmtime(cache_file)
        if (time.time() - file_time) < 86400:  # 24 hours in seconds
            try:
                with open(cache_file, 'r') as f:
                    stock_dict = json.load(f)
                with open(name_to_ticker_file, 'r') as f:
                    name_to_ticker = json.load(f)
                return stock_dict, name_to_ticker
            except Exception as e:
                st.warning(f"Error loading cache: {e}. Fetching fresh data.")
    
    # If cache doesn't exist or is old, fetch fresh data
    stock_dict = {}
    name_to_ticker = {}
    
    try:
        # 1. Fetch US Stocks from multiple sources
        # Yahoo Finance
        us_tickers_resp = requests.get(
            "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt",
            timeout=10
        )
        
        if us_tickers_resp.status_code == 200:
            us_tickers = us_tickers_resp.text.strip().split('\n')
            
            # Process in batches to avoid overloading
            batch_size = 100
            for i in range(0, len(us_tickers), batch_size):
                batch = us_tickers[i:i+batch_size]
                
                # Use ThreadPoolExecutor for parallel fetching
                with ThreadPoolExecutor(max_workers=5) as executor:
                    def fetch_ticker_info(ticker):
                        try:
                            stock = yf.Ticker(ticker)
                            info = stock.info
                            if 'shortName' in info and info['shortName']:
                                return ticker, info['shortName']
                            return None
                        except Exception:
                            return None
                    
                    results = executor.map(fetch_ticker_info, batch)
                    
                    for result in results:
                        if result:
                            ticker, name = result
                            stock_dict[ticker] = name
                            name_to_ticker[name.lower()] = ticker
                
                # Show progress
                st.session_state['progress'] = min(100, int((i + batch_size) / len(us_tickers) * 100))
                
        # 2. Fetch Indian stocks (NSE)
        try:
            nse_url = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            nse_resp = requests.get(nse_url, headers=headers, timeout=10)
            
            if nse_resp.status_code == 200:
                nse_data = pd.read_csv(pd.StringIO(nse_resp.text))
                for _, row in nse_data.iterrows():
                    try:
                        symbol = row['SYMBOL']
                        company_name = row['NAME OF COMPANY']
                        ticker = f"{symbol}.NS"
                        stock_dict[ticker] = company_name
                        name_to_ticker[company_name.lower()] = ticker
                    except Exception:
                        continue
        except Exception as e:
            st.warning(f"Error fetching NSE stocks: {e}")
            
            # Fallback to a smaller set of major Indian stocks
            major_indian_tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
                "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
                "ITC.NS", "ADANIENT.NS", "HCLTECH.NS", "WIPRO.NS", "AXISBANK.NS"
            ]
            
            for ticker in major_indian_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if 'shortName' in info and info['shortName']:
                        stock_dict[ticker] = info['shortName']
                        name_to_ticker[info['shortName'].lower()] = ticker
                except Exception:
                    continue
        
        # 3. Fetch Indian stocks (BSE)
        try:
            major_bse_tickers = [
                "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "HINDUNILVR.BO",
                "ICICIBANK.BO", "SBIN.BO", "BAJFINANCE.BO", "BHARTIARTL.BO", "KOTAKBANK.BO"
            ]
            
            for ticker in major_bse_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if 'shortName' in info and info['shortName']:
                        stock_dict[ticker] = info['shortName']
                        name_to_ticker[info['shortName'].lower()] = ticker
                except Exception:
                    continue
        except Exception as e:
            st.warning(f"Error fetching BSE stocks: {e}")
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(stock_dict, f)
            with open(name_to_ticker_file, 'w') as f:
                json.dump(name_to_ticker, f)
        except Exception as e:
            st.warning(f"Error saving cache: {e}")
            
        return stock_dict, name_to_ticker
    
    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        # Return a minimal set of example stocks as fallback
        fallback_stocks = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "AMZN": "Amazon.com, Inc.",
            "GOOGL": "Alphabet Inc.",
            "RELIANCE.NS": "Reliance Industries Limited",
            "TCS.NS": "Tata Consultancy Services Limited"
        }
        fallback_name_to_ticker = {
            "apple inc.": "AAPL",
            "microsoft corporation": "MSFT",
            "amazon.com, inc.": "AMZN",
            "alphabet inc.": "GOOGL",
            "reliance industries limited": "RELIANCE.NS",
            "tata consultancy services limited": "TCS.NS"
        }
        return fallback_stocks, fallback_name_to_ticker

# Calculate RSI (Relative Strength Index)
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]

# Calculate MACD (Moving Average Convergence Divergence)
def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    fast_ma = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ma = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ma - slow_ma
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    return macd_line.iloc[-1], signal_line.iloc[-1]

# Determine market trend
def determine_trend(hist):
    # Get short term (5-day) and medium term (20-day) moving averages
    if len(hist) >= 20:
        ma5 = hist['Close'].rolling(window=5).mean().iloc[-1]
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        current_price = hist['Close'].iloc[-1]
        
        # Determine price action over different periods
        one_day_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
        one_week_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100 if len(hist) >= 5 else 0
        one_month_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        
        # Determine the trend based on moving averages and price action
        if ma5 > ma20 and current_price > ma5:
            trend = "Strong Uptrend"
        elif ma5 > ma20:
            trend = "Moderate Uptrend"
        elif ma5 < ma20 and current_price < ma5:
            trend = "Strong Downtrend"
        elif ma5 < ma20:
            trend = "Moderate Downtrend"
        else:
            trend = "Sideways/Neutral"
            
        trend_data = {
            "trend": trend,
            "1d_change": f"{one_day_change:.2f}%",
            "1w_change": f"{one_week_change:.2f}%",
            "1m_change": f"{one_month_change:.2f}%",
            "ma5": ma5,
            "ma20": ma20
        }
        
        return trend_data
    else:
        return {"trend": "Insufficient Data", "1d_change": "N/A", "1w_change": "N/A", "1m_change": "N/A"}

# Function to get news from multiple sources including Indian sources
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_comprehensive_news(ticker):
    # Check cache first
    if ticker in st.session_state.news_cache:
        cache_time, news_items = st.session_state.news_cache[ticker]
        if (datetime.now() - cache_time).seconds < 1800:  # Cache valid for 30 minutes
            return news_items
    
    news_items = []
    
    # Determine if it's an Indian stock
    is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
    base_ticker = ticker.replace('.NS', '').replace('.BO', '')
    
    # 1. Yahoo Finance (Primary source)
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            for item in news[:5]:  # Get top 5 news items
                news_items.append({
                    "source": "Yahoo Finance",
                    "title": item.get('title', 'No title'),
                    "url": item.get('link', ''),
                    "published": datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d') if item.get('providerPublishTime') else 'Unknown',
                    "summary": item.get('summary', '')
                })
    except Exception as e:
        # Fallback to HTML scraping if API doesn't work
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_elements = soup.find_all('h3', {'class': 'Mb(5px)'})
            
            for element in news_elements[:3]:
                if element.a:
                    news_items.append({
                        "source": "Yahoo Finance",
                        "title": element.a.text,
                        "url": "https://finance.yahoo.com" + element.a.get('href', ''),
                        "published": "Recent",
                        "summary": ""
                    })
        except Exception:
            pass
    
    # 2. MarketWatch
    try:
        url = f"https://www.marketwatch.com/investing/stock/{base_ticker.lower()}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_elements = soup.select('.article__headline')
        
        for i, element in enumerate(news_elements[:3]):
            if i >= 3:  # Limit to 3 news items
                break
                
            title = element.text.strip()
            link = element.parent.get('href', '') if element.parent else ''
            
            # Only add if we have a title
            if title:
                news_items.append({
                    "source": "MarketWatch",
                    "title": title,
                    "url": link,
                    "published": "Recent",
                    "summary": ""
                })
    except Exception:
        pass
    
    # 3. Investing.com
    try:
        # For US stocks
        if not is_indian:
            url = f"https://www.investing.com/equities/{base_ticker.lower()}-news"
        else:
            # For Indian stocks - use a search
            url = f"https://www.investing.com/search/?q={base_ticker}"
            
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Different selectors for US and Indian stocks
        if not is_indian:
            news_elements = soup.select('.articleItem')
            for element in news_elements[:3]:
                title_el = element.select_one('.title')
                if title_el:
                    title = title_el.text.strip()
                    link = title_el.get('href', '')
                    if link and not link.startswith('http'):
                        link = f"https://www.investing.com{link}"
                        
                    # Get date if available
                    date_el = element.select_one('.date')
                    date = date_el.text.strip() if date_el else "Recent"
                    
                    news_items.append({
                        "source": "Investing.com",
                        "title": title,
                        "url": link,
                        "published": date,
                        "summary": ""
                    })
        else:
            # For Indian stocks, look in search results
            news_elements = soup.select('.js-article-item')
            for element in news_elements[:3]:
                title_el = element.select_one('.js-article-item-title')
                if title_el:
                    title = title_el.text.strip()
                    link = title_el.get('href', '')
                    if link and not link.startswith('http'):
                        link = f"https://www.investing.com{link}"
                        
                    news_items.append({
                        "source": "Investing.com",
                        "title": title,
                        "url": link,
                        "published": "Recent",
                        "summary": ""
                    })
    except Exception:
        pass
    
    # 4. For Indian stocks, try Indian financial news sources
    if is_indian:
        # Try MoneyControl
        try:
            url = f"https://www.moneycontrol.com/news/business/stocks/searchresult.php?q={base_ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_elements = soup.select('.common_search_list h2')
            
            for element in news_elements[:3]:
                if element.a:
                    title = element.a.text.strip()
                    link = element.a.get('href', '')
                    news_items.append({
                        "source": "MoneyControl",
                        "title": title,
                        "url": link,
                        "published": "Recent",
                        "summary": ""
                    })
        except Exception:
            pass
            
        # Try Economic Times
        try:
            url = f"https://economictimes.indiatimes.com/searchresult.cms?query={base_ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_elements = soup.select('.eachStory')
            
            for element in news_elements[:3]:
                title_el = element.select_one('h3')
                if title_el and title_el.a:
                    title = title_el.a.text.strip()
                    link = title_el.a.get('href', '')
                    if link and not link.startswith('http'):
                        link = f"https://economictimes.indiatimes.com{link}"
                        
                    date_el = element.select_one('.flt.date')
                    date = date_el.text.strip() if date_el else "Recent"
                    
                    news_items.append({
                        "source": "Economic Times",
                        "title": title,
                        "url": link,
                        "published": date,
                        "summary": ""
                    })
        except Exception:
            pass
    
    # If no news found from any source
    if not news_items:
        return "No recent news found."
    
    # Cache the results
    st.session_state.news_cache[ticker] = (datetime.now(), news_items)
    
    return news_items

# Function to summarize news with ML if available, or rule-based if not
def summarize_news(news_items):
    if isinstance(news_items, str):
        return news_items
    
    # Extract all news titles for analysis
    all_titles = [item["title"] for item in news_items if isinstance(item, dict) and "title" in item]
    all_summaries = [item.get("summary", "") for item in news_items if isinstance(item, dict) and "summary" in item]
    
    if not all_titles:
        return "No news titles available for analysis."
    
    # Combine titles for ML summarization
    combined_text = " ".join(all_titles)
    combined_text += " " + " ".join([s for s in all_summaries if s])
    
    # Try ML summarization if the model is available
    if st.session_state.ml_summarizer and len(combined_text) > 100:
        try:
            ml_summary = st.session_state.ml_summarizer(
                combined_text, 
                max_length=150, 
                min_length=30, 
                do_sample=False
            )
            
            if ml_summary and len(ml_summary) > 0:
                return ml_summary[0]['summary_text']
        except Exception as e:
            st.warning(f"ML summarization failed: {e}. Falling back to rule-based approach.")
    
    # Fall back to keyword-based sentiment analysis
    positive_keywords = ['rise', 'growth', 'profit', 'up', 'gain', 'positive', 'beat', 'bullish', 'surged', 'higher',
                         'increase', 'improved', 'upgrade', 'opportunity', 'outperform', 'strong', 'success']
    negative_keywords = ['drop', 'fall', 'decline', 'down', 'loss', 'negative', 'miss', 'bearish', 'plunged', 'lower',
                         'decrease', 'downgrade', 'risk', 'underperform', 'weak', 'concern', 'warn', 'crisis']
    
    positive_count = 0
    negative_count = 0
    
    for title in all_titles:
        title_lower = title.lower()
        positive_count += sum(1 for word in positive_keywords if word in title_lower)
        negative_count += sum(1 for word in negative_keywords if word in title_lower)
    
    # Extract key topics
    words = ' '.join(all_titles).lower()
    words = re.sub(r'[^\w\s]', '', words)
    word_list = words.split()
    
    # Remove common words
    common_words = ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'and', 'but', 'or', 'as', 'of', 'to', 'with', 'by']
    filtered_words = [w for w in word_list if w not in common_words and len(w) > 3]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Get top topics
    top_topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Determine overall sentiment
    if positive_count > negative_count * 1.5:
        sentiment = "Very Positive"
    elif positive_count > negative_count:
        sentiment = "Moderately Positive"
    elif negative_count > positive_count * 1.5:
        sentiment = "Very Negative"
    elif negative_count > positive_count:
        sentiment = "Moderately Negative"
    else:
        sentiment = "Neutral"
    
    # Generate a summary
    summary = f"News Sentiment: {sentiment}. "
    summary += f"Analysis of {len(news_items)} recent headlines revealed {positive_count} positive and {negative_count} negative indicators. "
    
    if top_topics:
        summary += f"Key topics mentioned: {', '.join([topic[0] for topic in top_topics])}. "
    
    # Add a recency note
    recent_dates = [item.get("published", "") for item in news_items if isinstance(item, dict)]
    if any("today" in date.lower() for date in recent_dates if date):
        summary += "Several headlines are from today, indicating active market interest."
    
    return summary

# Four types of analysis functions
def fundamental_analysis(info):
    analysis = {}
    
    # P/E Ratio Analysis
    pe_ratio = info.get('trailingPE')
    if pe_ratio:
        if pe_ratio < 15:
            analysis['pe'] = f"P/E ratio of {pe_ratio:.2f} suggests potential undervaluation compared to market average."
        elif pe_ratio > 30:
            analysis['pe'] = f"P/E ratio of {pe_ratio:.2f} indicates possible overvaluation compared to market average."
        else:
            analysis['pe'] = f"P/E ratio of {pe_ratio:.2f} appears reasonable and close to market average."
    else:
        analysis['pe'] = "P/E ratio data unavailable for analysis."
    
    # Price to Book Analysis
    pb_ratio = info.get('priceToBook')
    if pb_ratio:
        if pb_ratio < 1:
            analysis['pb'] = f"Price-to-Book ratio of {pb_ratio:.2f} suggests the stock may be undervalued relative to its assets."
        elif pb_ratio > 3:
            analysis['pb'] = f"Price-to-Book ratio of {pb_ratio:.2f} indicates the stock may be trading at a premium to its assets."
        else:
            analysis['pb'] = f"Price-to-Book ratio of {pb_ratio:.2f} is within a reasonable range."
    else:
        analysis['pb'] = "Price-to-Book data unavailable for analysis."
    
    # Dividend Analysis
    dividend_yield = info.get('dividendYield', 0)
    if dividend_yield:
        dividend_percentage = dividend_yield * 100
        if dividend_percentage > 4:
            analysis['dividend'] = f"High dividend yield of {dividend_percentage:.2f}% suggests good income potential."
        elif dividend_percentage > 0:
            analysis['dividend'] = f"Moderate dividend yield of {dividend_percentage:.2f}% provides some income."
        else:
            analysis['dividend'] = "No dividend offered, focusing solely on growth potential."
    else:
        analysis['dividend'] = "No dividend data available."
    
    # EPS Growth
    eps_ttm = info.get('trailingEps')
    forward_eps = info.get('forwardEps')
    
    if eps_ttm and forward_eps:
        eps_growth = ((forward_eps / eps_ttm) - 1) * 100
        if eps_growth > 20:
            analysis['eps_growth'] = f"Strong projected EPS growth of {eps_growth:.2f}% indicates excellent earnings potential."
        elif eps_growth > 10:
            analysis['eps_growth'] = f"Good projected EPS growth of {eps_growth:.2f}% suggests healthy earnings outlook."
        elif eps_growth > 0:
            analysis['eps_growth'] = f"Modest projected EPS growth of {eps_growth:.2f}% indicates stable earnings."
        else:
            analysis['eps_growth'] = f"Projected EPS decline of {abs(eps_growth):.2f}% suggests potential earnings challenges."
    else:
        analysis['eps_growth'] = "EPS growth data unavailable for analysis."
    
    return analysis

def technical_analysis(hist, rsi):
    analysis = {}

    # Get the latest RSI value
    # rsi_value = rsi.iloc[-1]
    # Get the latest RSI value from the Series
    rsi_value = rsi.iloc[-1]

# Now safely format it
    st.markdown(f"RSI of {rsi_value:.1f} indicates oversold conditions, suggesting potential reversal upward.")


    # RSI Analysis
    if rsi_value < 30:
        analysis['rsi'] = f"RSI of {rsi_value:.1f} indicates oversold conditions, suggesting potential reversal upward."
    elif rsi_value > 70:
        analysis['rsi'] = f"RSI of {rsi_value:.1f} suggests overbought conditions, potential for downward correction."
    else:
        analysis['rsi'] = f"RSI of {rsi_value:.1f} shows neutral momentum, neither overbought nor oversold."

    if len(hist) >= 50:
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None
        current_price = hist['Close'].iloc[-1]
        
        if ma200:
            if current_price > ma50 and current_price > ma200 and ma50 > ma200:
                analysis['ma'] = "Price above both 50-day and 200-day moving averages with 50-day above 200-day (Golden Cross) - bullish signal."
            elif current_price < ma50 and current_price < ma200 and ma50 < ma200:
                analysis['ma'] = "Price below both 50-day and 200-day moving averages with 50-day below 200-day (Death Cross) - bearish signal."
            else:
                analysis['ma'] = "Mixed signals from moving averages, no clear trend direction."
        else:
            if current_price > ma50:
                analysis['ma'] = "Price above 50-day moving average suggests short-term uptrend."
            else:
                analysis['ma'] = "Price below 50-day moving average suggests short-term downtrend."
    else:
        analysis['ma'] = "Insufficient historical data for moving average analysis."
    
    # Volume Analysis
    if len(hist) >= 20:
        avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
        recent_volume = hist['Volume'].iloc[-1]
        
        if recent_volume > avg_volume * 1.5:
            analysis['volume'] = "Above-average trading volume suggests strong investor interest and potential trend confirmation."
        elif recent_volume < avg_volume * 0.5:
            analysis['volume'] = "Below-average trading volume indicates lack of investor interest or conviction."
        else:
            analysis['volume'] = "Normal trading volume, neither particularly strong nor weak."
    else:
        analysis['volume'] = "Insufficient historical data for volume analysis."
    
    # MACD Analysis
    if len(hist) >= 26:
        macd_line, signal_line = calculate_macd(hist['Close'])
        
        if macd_line > signal_line and macd_line > 0:
            analysis['macd'] = "MACD above signal line and zero line - strong bullish momentum."
        elif macd_line > signal_line and macd_line < 0:
            analysis['macd'] = "MACD above signal line but below zero line - potential bullish reversal from downtrend."
        elif macd_line < signal_line and macd_line > 0:
            analysis['macd'] = "MACD below signal line but above zero line - potential bearish reversal from uptrend."
        else:
            analysis['macd'] = "MACD below signal line and zero line - strong bearish momentum."
    else:
        analysis['macd'] = "Insufficient historical data for MACD analysis."
    
    return analysis

def sentiment_analysis(news_summary, info, hist):
    analysis = {}
    
    # News Sentiment
    if isinstance(news_summary, str) and "News Sentiment:" in news_summary:
        sentiment_match = re.search(r"News Sentiment: (.*?)\.", news_summary)
        if sentiment_match:
            sentiment = sentiment_match.group(1)
            analysis['news'] = f"Overall news sentiment is {sentiment.lower()}."
    else:
        analysis['news'] = "Unable to determine news sentiment due to insufficient data."
    
    # Analyst Recommendations
    recommendations = info.get('recommendationMean')
    if recommendations:
        if recommendations < 2:
            analysis['analyst'] = f"Strong analyst buy rating of {recommendations:.1f} (scale 1-5, where 1 is strong buy)."
        elif recommendations < 2.5:
            analysis['analyst'] = f"Moderate analyst buy rating of {recommendations:.1f} (scale 1-5)."
        elif recommendations < 3.5:
            analysis['analyst'] = f"Neutral analyst rating of {recommendations:.1f} (scale 1-5)."
        else:
            analysis['analyst'] = f"Analyst sell rating of {recommendations:.1f} (scale 1-5)."
    else:
        analysis['analyst'] = "No analyst recommendation data available."
    
    # Short Interest
    short_percentage = info.get('shortPercentOfFloat')
    if short_percentage:
        short_percentage = short_percentage * 100
        if short_percentage > 20:
            analysis['short_interest'] = f"Very high short interest of {short_percentage:.2f}% indicates significant bearish sentiment."
        elif short_percentage > 10:
            analysis['short_interest'] = f"High short interest of {short_percentage:.2f}% suggests considerable bearish sentiment."
        elif short_percentage > 5:
            analysis['short_interest'] = f"Moderate short interest of {short_percentage:.2f}%."
        else:
            analysis['short_interest'] = f"Low short interest of {short_percentage:.2f}% indicates limited bearish pressure."
    else:
        analysis['short_interest'] = "Short interest data unavailable."
    
    # Price momentum (last month)
    if len(hist) >= 20:
        start_price = hist['Close'].iloc[-20]
        end_price = hist['Close'].iloc[-1]
        momentum = ((end_price / start_price) - 1) * 100
        
        if momentum > 10:
            analysis['momentum'] = f"Strong positive price momentum of {momentum:.2f}% over the past month."
        elif momentum > 5:
            analysis['momentum'] = f"Moderately positive price momentum of {momentum:.2f}% over the past month."
        elif momentum > -5:
            analysis['momentum'] = f"Relatively flat price momentum of {momentum:.2f}% over the past month."
        elif momentum > -10:
            analysis['momentum'] = f"Moderately negative price momentum of {abs(momentum):.2f}% over the past month."
        else:
            analysis['momentum'] = f"Strong negative price momentum of {abs(momentum):.2f}% over the past month."
    else:
        analysis['momentum'] = "Insufficient historical data for momentum analysis."
    
    return analysis

def risk_analysis(info, hist):
    analysis = {}
    
    # Beta Analysis
    beta = info.get('beta')
    if beta:
        if beta > 1.5:
            analysis['beta'] = f"High beta of {beta:.2f} indicates significant volatility compared to the market."
        elif beta > 1:
            analysis['beta'] = f"Beta of {beta:.2f} suggests slightly higher volatility than the market."
        elif beta > 0.5:
            analysis['beta'] = f"Beta of {beta:.2f} indicates slightly lower volatility than the market."
        else:
            analysis['beta'] = f"Low beta of {beta:.2f} suggests significantly lower volatility than the market."
    else:
        analysis['beta'] = "Beta data unavailable for analysis."
    
    # Historical Volatility
    if len(hist) >= 20:
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        if volatility > 50:
            analysis['volatility'] = f"Extremely high volatility of {volatility:.2f}% indicates substantial price fluctuations."
        elif volatility > 30:
            analysis['volatility'] = f"High volatility of {volatility:.2f}% suggests significant price swings."
        elif volatility > 15:
            analysis['volatility'] = f"Moderate volatility of {volatility:.2f}%, typical for average stocks."
        else:
            analysis['volatility'] = f"Low volatility of {volatility:.2f}% indicates relatively stable price action."
    else:
        analysis['volatility'] = "Insufficient historical data for volatility analysis."
    
    # Debt-to-Equity Analysis
    debt_to_equity = info.get('debtToEquity')
    if debt_to_equity:
        if debt_to_equity > 2:
            analysis['debt'] = f"High debt-to-equity ratio of {debt_to_equity:.2f} indicates significant financial leverage."
        elif debt_to_equity > 1:
            analysis['debt'] = f"Moderate debt-to-equity ratio of {debt_to_equity:.2f}."
        else:
            analysis['debt'] = f"Low debt-to-equity ratio of {debt_to_equity:.2f} suggests conservative financial management."
    else:
        analysis['debt'] = "Debt-to-equity data unavailable for analysis."
    
    # Liquidity Analysis (Average Daily Volume)
    if len(hist) >= 20:
        avg_daily_volume = hist['Volume'].mean()
        
        if avg_daily_volume > 5000000:
            analysis['liquidity'] = f"Very high average daily volume of {avg_daily_volume:.0f} shares indicates excellent liquidity."
        elif avg_daily_volume > 1000000:
            analysis['liquidity'] = f"Good average daily volume of {avg_daily_volume:.0f} shares suggests sufficient liquidity."
        elif avg_daily_volume > 100000:
            analysis['liquidity'] = f"Moderate average daily volume of {avg_daily_volume:.0f} shares."
        else:
            analysis['liquidity'] = f"Low average daily volume of {avg_daily_volume:.0f} shares indicates potential liquidity concerns."
    else:
        analysis['liquidity'] = "Insufficient historical data for liquidity analysis."
    
    return analysis

# Generate final recommendation based on comprehensive analysis
def generate_recommendation(fundamental, technical, sentiment, risk, trend_data):
    # Assign weights to different types of analysis
    weights = {
        'fundamental': 0.3,
        'technical': 0.3,
        'sentiment': 0.25,
        'risk': 0.15
    }
    
    # Score each analysis from -5 to 5
    scores = {
        'fundamental': 0,
        'technical': 0,
        'sentiment': 0,
        'risk': 0
    }
    
    # Score fundamental analysis
    for key, value in fundamental.items():
        value_lower = value.lower()
        
        if key == 'pe':
            if "potential undervaluation" in value_lower:
                scores['fundamental'] += 2
            elif "possible overvaluation" in value_lower:
                scores['fundamental'] -= 2
                
        if key == 'pb':
            if "may be undervalued" in value_lower:
                scores['fundamental'] += 1.5
            elif "may be trading at a premium" in value_lower:
                scores['fundamental'] -= 1
                
        if key == 'dividend':
            if "high dividend yield" in value_lower:
                scores['fundamental'] += 1
            elif "moderate dividend yield" in value_lower:
                scores['fundamental'] += 0.5
                
        if key == 'eps_growth':
            if "strong projected" in value_lower:
                scores['fundamental'] += 2
            elif "good projected" in value_lower:
                scores['fundamental'] += 1
            elif "modest projected" in value_lower:
                scores['fundamental'] += 0.5
            elif "projected eps decline" in value_lower:
                scores['fundamental'] -= 1.5
    
    # Normalize fundamental score
    scores['fundamental'] = max(min(scores['fundamental'], 5), -5)
    
    # Score technical analysis
    for key, value in technical.items():
        value_lower = value.lower()
        
        if key == 'rsi':
            if "oversold" in value_lower:
                scores['technical'] += 1.5
            elif "overbought" in value_lower:
                scores['technical'] -= 1.5
                
        if key == 'ma':
            if "golden cross" in value_lower or "uptrend" in value_lower:
                scores['technical'] += 2
            elif "death cross" in value_lower or "downtrend" in value_lower:
                scores['technical'] -= 2
                
        if key == 'volume':
            if "above-average" in value_lower:
                # Check trend direction to determine if volume confirms
                if 'trend' in trend_data and ('uptrend' in trend_data['trend'].lower() or 'downtrend' in trend_data['trend'].lower()):
                    scores['technical'] += 1
            elif "below-average" in value_lower:
                scores['technical'] -= 0.5
                
        if key == 'macd':
            if "strong bullish" in value_lower:
                scores['technical'] += 1.5
            elif "potential bullish" in value_lower:
                scores['technical'] += 0.5
            elif "potential bearish" in value_lower:
                scores['technical'] -= 0.5
            elif "strong bearish" in value_lower:
                scores['technical'] -= 1.5
    
    # Normalize technical score
    scores['technical'] = max(min(scores['technical'], 5), -5)
    
    # Score sentiment analysis
    for key, value in sentiment.items():
        value_lower = value.lower()
        
        if key == 'news':
            if "very positive" in value_lower:
                scores['sentiment'] += 2
            elif "moderately positive" in value_lower:
                scores['sentiment'] += 1
            elif "very negative" in value_lower:
                scores['sentiment'] -= 2
            elif "moderately negative" in value_lower:
                scores['sentiment'] -= 1
                
        if key == 'analyst':
            if "strong analyst buy" in value_lower:
                scores['sentiment'] += 2
            elif "moderate analyst buy" in value_lower:
                scores['sentiment'] += 1
            elif "analyst sell" in value_lower:
                scores['sentiment'] -= 2
                
        if key == 'short_interest':
            if "very high short interest" in value_lower:
                scores['sentiment'] -= 1.5
            elif "high short interest" in value_lower:
                scores['sentiment'] -= 1
            elif "low short interest" in value_lower:
                scores['sentiment'] += 0.5
                
        if key == 'momentum':
            if "strong positive" in value_lower:
                scores['sentiment'] += 1.5
            elif "moderately positive" in value_lower:
                scores['sentiment'] += 0.75
            elif "moderately negative" in value_lower:
                scores['sentiment'] -= 0.75
            elif "strong negative" in value_lower:
                scores['sentiment'] -= 1.5
    
    # Normalize sentiment score
    scores['sentiment'] = max(min(scores['sentiment'], 5), -5)
    
    # Score risk analysis (higher risk lowers score)
    for key, value in risk.items():
        value_lower = value.lower()
        
        if key == 'beta':
            if "high beta" in value_lower:
                scores['risk'] -= 1.5
            elif "slightly higher volatility" in value_lower:
                scores['risk'] -= 0.5
            elif "slightly lower volatility" in value_lower:
                scores['risk'] += 0.5
            elif "low beta" in value_lower:
                scores['risk'] += 1
                
        if key == 'volatility':
            if "extremely high volatility" in value_lower:
                scores['risk'] -= 2
            elif "high volatility" in value_lower:
                scores['risk'] -= 1
            elif "low volatility" in value_lower:
                scores['risk'] += 1
                
        if key == 'debt':
            if "high debt-to-equity" in value_lower:
                scores['risk'] -= 1.5
            elif "low debt-to-equity" in value_lower:
                scores['risk'] += 1
                
        if key == 'liquidity':
            if "very high average daily volume" in value_lower or "good average daily volume" in value_lower:
                scores['risk'] += 1
            elif "low average daily volume" in value_lower:
                scores['risk'] -= 1.5
    
    # Normalize risk score
    scores['risk'] = max(min(scores['risk'], 5), -5)
    
    # Calculate weighted total score
    total_score = 0
    for category, score in scores.items():
        total_score += score * weights[category]
    
    # Generate recommendation based on total score
    recommended_action = ""
    recommendation_class = ""
    recommendation_details = ""
    
    if total_score > 3:
        recommended_action = "Strong Buy"
        recommendation_class = "recommendation-buy"
        recommendation_details = "Excellent overall profile with strong fundamentals, technicals, and positive sentiment."
    elif total_score > 1.5:
        recommended_action = "Buy"
        recommendation_class = "recommendation-buy"
        recommendation_details = "Good overall profile with positive indicators across multiple analysis categories."
    elif total_score > 0:
        recommended_action = "Mild Buy"
        recommendation_class = "recommendation-buy"
        recommendation_details = "Slightly positive profile, consider buying with caution or smaller position."
    elif total_score > -1.5:
        recommended_action = "Hold"
        recommendation_class = "recommendation-hold"
        recommendation_details = "Mixed signals suggest maintaining current positions without adding more."
    elif total_score > -3:
        recommended_action = "Sell"
        recommendation_class = "recommendation-sell"
        recommendation_details = "Negative outlook across multiple categories suggests reducing exposure."
    else:
        recommended_action = "Strong Sell"
        recommendation_class = "recommendation-sell"
        recommendation_details = "Significant negative indicators across most analysis categories."
    
    # Add additional context based on trend
    if 'trend' in trend_data:
        trend = trend_data['trend'].lower()
        if 'strong uptrend' in trend and total_score > 0:
            recommendation_details += " Stock is in a strong uptrend, which supports the buying recommendation."
        elif 'strong downtrend' in trend and total_score < 0:
            recommendation_details += " Stock is in a strong downtrend, which supports the selling recommendation."
        elif 'strong uptrend' in trend and total_score < 0:
            recommendation_details += " Despite the stock being in a strong uptrend, other factors suggest caution."
        elif 'strong downtrend' in trend and total_score > 0:
            recommendation_details += " Despite the stock being in a strong downtrend, other factors show potential."
    
    # Format scores for display
    scores_display = {key: f"{value:.1f}" for key, value in scores.items()}
    
    return {
        'action': recommended_action,
        'class': recommendation_class,
        'details': recommendation_details,
        'scores': scores_display,
        'total_score': f"{total_score:.2f}"
    }

# Layout for the dashboard
def main():
    # Title and description
    st.title("📈 Enhanced Stock Analysis Dashboard")
    st.markdown("""
    This dashboard provides comprehensive stock analysis for US and Indian markets. 
    Enter a stock ticker or company name to get started.
    """)
    
    # Search functionality for stock tickers
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        stock_dict, name_to_ticker = load_stock_data()
        
        # Convert all keys to lowercase for case-insensitive search
        search_options = list(stock_dict.keys()) + list(name_to_ticker.keys())
        
        with col1:
            search_query = st.text_input("Enter stock ticker or company name", placeholder="e.g., AAPL, Microsoft, RELIANCE.NS").strip()
        
        with col2:
            search_button = st.button("Analyze Stock", use_container_width=True)
        
        if search_query and search_button:
            # First check if it's a direct ticker match
            if search_query.upper() in stock_dict:
                selected_ticker = search_query.upper()
            # Then check if it could be a company name
            elif search_query.lower() in name_to_ticker:
                selected_ticker = name_to_ticker[search_query.lower()]
            # Try partial matching - find the first partial match in tickers
            else:
                ticker_matches = [ticker for ticker in stock_dict.keys() 
                                if search_query.upper() in ticker]
                name_matches = [ticker for name, ticker in name_to_ticker.items() 
                               if search_query.lower() in name]
                
                if ticker_matches:
                    selected_ticker = ticker_matches[0]
                elif name_matches:
                    selected_ticker = name_matches[0]
                else:
                    st.error(f"No match found for '{search_query}'. Please enter a valid ticker or company name.")
                    st.stop()
            
            st.success(f"Analyzing: {selected_ticker} - {stock_dict.get(selected_ticker, 'Unknown Company')}")
            
            # Create placeholder for stock analysis
            analysis_placeholder = st.empty()
            
            with analysis_placeholder.container():
                with st.spinner("Fetching stock data and analyzing..."):
                    # Get stock data
                    try:
                        # Check cache first
                        if selected_ticker in st.session_state.stock_data_cache:
                            stock = st.session_state.stock_data_cache[selected_ticker]["stock_obj"]
                            hist = st.session_state.stock_data_cache[selected_ticker]["hist"]
                            info = st.session_state.stock_data_cache[selected_ticker]["info"]
                        else:
                            stock = yf.Ticker(selected_ticker)
                            hist = stock.history(period="1y")
                            info = stock.info
                            
                            # Cache the data
                            st.session_state.stock_data_cache[selected_ticker] = {
                                "stock_obj": stock,
                                "hist": hist,
                                "info": info,
                                "timestamp": datetime.now()
                            }
                            
                        # Calculate technical indicators
                        if not hist.empty:
                            rsi = calculate_rsi(hist['Close'])
                            macd_line, signal_line = calculate_macd(hist['Close'])
                            trend_data = determine_trend(hist)
                        else:
                            st.error(f"No historical data available for {selected_ticker}")
                            st.stop()
                            
                        # Get news and summarize
                        news_items = get_comprehensive_news(selected_ticker)
                        news_summary = summarize_news(news_items)
                        
                        # Perform all analyses
                        fundamental = fundamental_analysis(info)
                        technical = technical_analysis(hist, rsi)
                        sentiment = sentiment_analysis(news_summary, info, hist)
                        risk = risk_analysis(info, hist)
                        
                        # Generate final recommendation
                        recommendation = generate_recommendation(
                            fundamental, technical, sentiment, risk, trend_data
                        )
                        
                        # Begin displaying the dashboard
                        tabs = st.tabs(["Overview", "Detailed Analysis", "Charts", "News", "Financial Data"])
                        
                        # Overview Tab
                        with tabs[0]:
                            # First row: Stock info and trends
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                st.subheader(f"{stock_dict.get(selected_ticker, info.get('shortName', selected_ticker))}")
                                current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', 'N/A')
                                
                                if isinstance(current_price, (int, float)):
                                    prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
                                    change = current_price - prev_close
                                    change_pct = (change / prev_close) * 100 if prev_close else 0
                                    
                                    price_color = "green" if change >= 0 else "red"
                                    change_sign = "+" if change >= 0 else ""
                                    
                                    st.markdown(f"""
                                    <div style="display: flex; align-items: baseline;">
                                        <h1 style="margin: 0; padding: 0;">{current_price:.2f}</h1>
                                        <h3 style="margin: 0; padding-left: 10px; color: {price_color};">{change_sign}{change:.2f} ({change_sign}{change_pct:.2f}%)</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<h1>{current_price}</h1>", unsafe_allow_html=True)
                                
                                # Add exchange info and currency
                                exchange = info.get('exchange', 'Unknown Exchange')
                                currency = info.get('currency', 'USD')
                                st.text(f"{exchange} • {currency}")
                            
                            with col2:
                                st.subheader("Market Trend")
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{trend_data['trend']}</h3>
                                    <p>Daily: {trend_data['1d_change']} • Weekly: {trend_data['1w_change']} • Monthly: {trend_data['1m_change']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Technical strength indicators
                                with col3:
                                    st.subheader("Recommendation")
                                    # Make sure 'recommendation' contains the expected keys and values
                                    recommendation_class = recommendation.get('class', 'default-class')  # Default if not found
                                    recommendation_action = recommendation.get('action', 'No recommendation available')  # Default message if not found
                                    recommendation_score = recommendation.get('total_score', 0)  # Default score if not found

                                    # Display the recommendation card
                                    st.markdown(f"""
                                    <div class="metric-card" style="text-align: center;">
                                        <div class="{recommendation_class}">{recommendation_action}</div>
                                        <p>Score: {recommendation_score}/5</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                            
                            with col3:
                                st.subheader("Recommendation")
                                st.markdown(f"""
                                <div class="metric-card" style="text-align: center;">
                                    <div class="{recommendation['class']}">{recommendation['action']}</div>
                                    <p>Score: {recommendation['total_score']}/5</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Second row: Analysis summary
                            st.subheader("Analysis Summary")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Fundamental Analysis</h4>
                                    <p>{fundamental.get('pe', 'N/A')}</p>
                                    <p>{fundamental.get('eps_growth', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Technical Analysis</h4>
                                    <p>{technical.get('rsi', 'N/A')}</p>
                                    <p>{technical.get('ma', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Sentiment Analysis</h4>
                                    <p>{sentiment.get('news', 'N/A')}</p>
                                    <p>{sentiment.get('analyst', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Risk Assessment</h4>
                                    <p>{risk.get('volatility', 'N/A')}</p>
                                    <p>{risk.get('beta', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Third row: Recommendation details
                            st.subheader("Investment Recommendation")
                            st.markdown(f"""
                            <div class="metric-card">
                                <p>{recommendation['details']}</p>
                                <p>Analysis Scores - Fundamental: {recommendation['scores']['fundamental']}, 
                                Technical: {recommendation['scores']['technical']}, 
                                Sentiment: {recommendation['scores']['sentiment']}, 
                                Risk: {recommendation['scores']['risk']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed Analysis Tab
                        with tabs[1]:
                            st.subheader("Comprehensive Stock Analysis")
                            
                            # Create four columns for the four types of analysis
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                with st.expander("⚖️ Fundamental Analysis", expanded=True):
                                    for key, value in fundamental.items():
                                        st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")
                                
                                with st.expander("📊 Technical Analysis", expanded=True):
                                    for key, value in technical.items():
                                        st.markdown(f"- **{key.upper()}**: {value}")
                            
                            with col2:
                                with st.expander("🧠 Sentiment Analysis", expanded=True):
                                    for key, value in sentiment.items():
                                        st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")
                                
                                with st.expander("⚠️ Risk Analysis", expanded=True):
                                    for key, value in risk.items():
                                        st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")
                            
                            # Additional details for serious investors
                            st.subheader("Additional Insights")
                            
                            # Financial Health
                            fin_col1, fin_col2 = st.columns(2)
                            
                            with fin_col1:
                                # Get some key financial metrics
                                profit_margins = info.get('profitMargins', 'N/A')
                                if isinstance(profit_margins, (int, float)):
                                    profit_margins = f"{profit_margins * 100:.2f}%"
                                    
                                operating_margins = info.get('operatingMargins', 'N/A')
                                if isinstance(operating_margins, (int, float)):
                                    operating_margins = f"{operating_margins * 100:.2f}%"
                                    
                                return_on_equity = info.get('returnOnEquity', 'N/A')
                                if isinstance(return_on_equity, (int, float)):
                                    return_on_equity = f"{return_on_equity * 100:.2f}%"
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Profitability</h4>
                                    <p>Profit Margin: {profit_margins}</p>
                                    <p>Operating Margin: {operating_margins}</p>
                                    <p>Return on Equity: {return_on_equity}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with fin_col2:
                                # Valuation metrics
                                pb_ratio = info.get('priceToBook', 'N/A')
                                pe_ratio = info.get('trailingPE', 'N/A')
                                peg_ratio = info.get('pegRatio', 'N/A')
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Valuation Metrics</h4>
                                    <p>P/E Ratio: {pe_ratio if isinstance(pe_ratio, str) else f"{pe_ratio:.2f}"}</p>
                                    <p>P/B Ratio: {pb_ratio if isinstance(pb_ratio, str) else f"{pb_ratio:.2f}"}</p>
                                    <p>PEG Ratio: {peg_ratio if isinstance(peg_ratio, str) else f"{peg_ratio:.2f}"}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Growth metrics
                            growth_col1, growth_col2 = st.columns(2)
                            
                            with growth_col1:
                                # Revenue and earnings growth
                                revenue_growth = info.get('revenueGrowth', 'N/A')
                                if isinstance(revenue_growth, (int, float)):
                                    revenue_growth = f"{revenue_growth * 100:.2f}%"
                                    
                                earnings_growth = info.get('earningsGrowth', 'N/A')
                                if isinstance(earnings_growth, (int, float)):
                                    earnings_growth = f"{earnings_growth * 100:.2f}%"
                                
                                st.markdown(f"""<div class="metric-card">
                                    <h4>Growth Metrics</h4>
                                    <p>Revenue Growth: {revenue_growth}</p>
                                    <p>Earnings Growth: {earnings_growth}</p>
                                    <p>Industry Position: {info.get('industryPosition', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with growth_col2:
                                # Future estimates
                                eps_next_quarter = info.get('epsForward', 'N/A')
                                eps_next_year = info.get('epsNextYear', 'N/A')
                                target_price = info.get('targetMeanPrice', 'N/A')
                                
                                current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', 'N/A')
                                if isinstance(target_price, (int, float)) and isinstance(current_price, (int, float)):
                                    target_upside = ((target_price / current_price) - 1) * 100
                                    target_text = f"{target_price:.2f} ({target_upside:.2f}% potential)"
                                else:
                                    target_text = str(target_price)
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Future Estimates</h4>
                                    <p>EPS Next Quarter: {eps_next_quarter if isinstance(eps_next_quarter, str) else f"${eps_next_quarter:.2f}"}</p>
                                    <p>EPS Next Year: {eps_next_year if isinstance(eps_next_year, str) else f"${eps_next_year:.2f}"}</p>
                                    <p>Analyst Target Price: {target_text}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Charts Tab
                        with tabs[2]:
                            st.subheader("Price Charts & Technical Indicators")
                            
                            # Date range selector
                            date_options = {
                                "1M": 30,
                                "3M": 90,
                                "6M": 180,
                                "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
                                "1Y": 365,
                                "MAX": len(hist)
                            }
                            
                            date_range = st.select_slider(
                                "Select Time Period",
                                options=list(date_options.keys()),
                                value="6M"
                            )
                            
                            days_to_show = min(date_options[date_range], len(hist))
                            chart_data = hist.iloc[-days_to_show:]
                            
                            # Chart type selector
                            chart_type = st.radio(
                                "Select Chart Type",
                                ["Candlestick", "Line", "OHLC"],
                                horizontal=True
                            )
                            
                            # Render the selected chart
                            fig = go.Figure()
                            
                            if chart_type == "Candlestick":
                                fig.add_trace(go.Candlestick(
                                    x=chart_data.index,
                                    open=chart_data['Open'],
                                    high=chart_data['High'],
                                    low=chart_data['Low'],
                                    close=chart_data['Close'],
                                    name="Price"
                                ))
                            elif chart_type == "Line":
                                fig.add_trace(go.Scatter(
                                    x=chart_data.index,
                                    y=chart_data['Close'],
                                    mode='lines',
                                    name="Close Price",
                                    line=dict(color='blue', width=2)
                                ))
                            else:  # OHLC
                                fig.add_trace(go.Ohlc(
                                    x=chart_data.index,
                                    open=chart_data['Open'],
                                    high=chart_data['High'],
                                    low=chart_data['Low'],
                                    close=chart_data['Close'],
                                    name="Price"
                                ))
                            
                            # Add volume as a bar chart on secondary y-axis
                            fig.add_trace(go.Bar(
                                x=chart_data.index,
                                y=chart_data['Volume'],
                                name="Volume",
                                yaxis="y2",
                                marker=dict(color='rgba(200, 200, 200, 0.5)')
                            ))
                            
                            # Add moving averages
                            ma_periods = [20, 50, 200]
                            ma_colors = ['orange', 'green', 'red']
                            
                            for period, color in zip(ma_periods, ma_colors):
                                if len(chart_data) >= period:
                                    ma = chart_data['Close'].rolling(window=period).mean()
                                    fig.add_trace(go.Scatter(
                                        x=chart_data.index,
                                        y=ma,
                                        mode='lines',
                                        name=f"{period}-day MA",
                                        line=dict(color=color, width=1)
                                    ))
                            
                            # Update layout
                            fig.update_layout(
                                title=f"{selected_ticker} Price Chart",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                height=600,
                                xaxis_rangeslider_visible=False,
                                yaxis2=dict(
                                    title="Volume",
                                    overlaying="y",
                                    side="right",
                                    showgrid=False
                                ),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            # Show the chart
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Technical indicators in separate charts
                            st.subheader("Technical Indicators")
                            indicator_cols = st.columns(2)
                            
                            with indicator_cols[0]:
                                # RSI Chart
                                rsi_data = calculate_rsi(chart_data['Close'])
                                
                                fig_rsi = go.Figure()
                                fig_rsi.add_trace(go.Scatter(
                                    x = chart_data.index[-len(rsi_data):],
                                    y=rsi_data,
                                    mode='lines',
                                    name="RSI",
                                    line=dict(color='purple', width=2)
                                ))
                                
                                # Add overbought/oversold lines
                                fig_rsi.add_shape(
                                    type="line",
                                    x0=chart_data.index[-len(rsi_data)],
                                    y0=70,
                                    x1=chart_data.index[-1],
                                    y1=70,
                                    line=dict(color="red", width=1, dash="dash")
                                )
                                
                                fig_rsi.add_shape(
                                    type="line",
                                    x0=chart_data.index[-len(rsi_data)],
                                    y0=30,
                                    x1=chart_data.index[-1],
                                    y1=30,
                                    line=dict(color="green", width=1, dash="dash")
                                )
                                
                                fig_rsi.update_layout(
                                    title="Relative Strength Index (RSI)",
                                    xaxis_title="Date",
                                    yaxis_title="RSI",
                                    height=300,
                                    yaxis=dict(range=[0, 100]),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig_rsi, use_container_width=True)
                            
                            with indicator_cols[1]:
                                # MACD Chart
                                if len(chart_data) >= 26:
                                    macd_line, signal_line = calculate_macd(chart_data['Close'])
                                    histogram = macd_line - signal_line

                                    fig_macd = go.Figure()

                                    # Function to get appropriate x-axis based on y values
                                    def get_x_axis(y_values):
                                        if isinstance(y_values, (np.ndarray, pd.Series, list)):
                                            return chart_data.index[-len(y_values):]
                                        else:
                                            return chart_data.index[-1:]

                                    # Wrap y values if they're scalar
                                    def wrap(y_values):
                                        return y_values if isinstance(y_values, (np.ndarray, pd.Series, list)) else [y_values]

                                    # MACD Line
                                    fig_macd.add_trace(go.Scatter(
                                        x=get_x_axis(macd_line),
                                        y=wrap(macd_line),
                                        mode='lines',
                                        name="MACD",
                                        line=dict(color='blue', width=2)
                                    ))

                                    # Signal Line
                                    fig_macd.add_trace(go.Scatter(
                                        x=get_x_axis(signal_line),
                                        y=wrap(signal_line),
                                        mode='lines',
                                        name="Signal",
                                        line=dict(color='red', width=1)
                                    ))

                                    # Histogram
                                    hist_vals = wrap(histogram)
                                    colors = ['green' if val >= 0 else 'red' for val in hist_vals]
                                    fig_macd.add_trace(go.Bar(
                                        x=get_x_axis(hist_vals),
                                        y=hist_vals,
                                        name="Histogram",
                                        marker_color=colors
                                    ))

                                    fig_macd.update_layout(
                                        title="MACD (Moving Average Convergence Divergence)",
                                        xaxis_title="Date",
                                        yaxis_title="Value",
                                        height=300,
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                        )
                                    )

                                    st.plotly_chart(fig_macd, use_container_width=True)

                        # News Tab
                        with tabs[3]:
                            st.subheader("Latest News & Analysis")
                            
                            # Display news summary
                            if news_summary:
                                st.markdown("### News Summary")
                                st.markdown(f"<div class='news-summary'>{news_summary}</div>", unsafe_allow_html=True)
                            
                            # Display individual news items
                            if news_items:
                                st.markdown("### Recent News Articles")
                                
                                for i, news in enumerate(news_items):
                                    with st.expander(f"{news['title']} - {news['publisher']}"):
                                        st.markdown(f"**Published:** {news['publishedDate']}")
                                        st.markdown(news['summary'])
                                        st.markdown(f"[Read full article]({news['link']})")
                            else:
                                st.info("No recent news available for this stock.")
                        
                        # Financial Data Tab
                        with tabs[4]:
                            st.subheader("Financial Statements & Ratios")
                            
                            # Create tabs for different financial statements
                            fin_tabs = st.tabs(["Key Ratios", "Income Statement", "Balance Sheet", "Cash Flow"])
                            
                            with fin_tabs[0]:
                                # Display key financial ratios in a table
                                st.markdown("### Key Financial Ratios")
                                
                                ratio_data = {
                                    "Valuation Ratios": {
                                        "P/E Ratio": info.get('trailingPE', 'N/A'),
                                        "Forward P/E": info.get('forwardPE', 'N/A'),
                                        "PEG Ratio": info.get('pegRatio', 'N/A'),
                                        "Price/Sales": info.get('priceToSalesTrailing12Months', 'N/A'),
                                        "Price/Book": info.get('priceToBook', 'N/A'),
                                        "Enterprise Value/EBITDA": info.get('enterpriseToEbitda', 'N/A')
                                    },
                                    "Profitability Ratios": {
                                        "Profit Margin": info.get('profitMargins', 'N/A'),
                                        "Operating Margin": info.get('operatingMargins', 'N/A'),
                                        "Return on Assets": info.get('returnOnAssets', 'N/A'),
                                        "Return on Equity": info.get('returnOnEquity', 'N/A')
                                    },
                                    "Dividend Metrics": {
                                        "Dividend Yield": info.get('dividendYield', 'N/A'),
                                        "Dividend Rate": info.get('dividendRate', 'N/A'),
                                        "Payout Ratio": info.get('payoutRatio', 'N/A'),
                                        "Dividend Date": info.get('dividendDate', 'N/A')
                                    },
                                    "Financial Health": {
                                        "Current Ratio": info.get('currentRatio', 'N/A'),
                                        "Debt to Equity": info.get('debtToEquity', 'N/A'),
                                        "Quick Ratio": info.get('quickRatio', 'N/A'),
                                        "Total Debt/Total Equity": info.get('totalDebt/totalEquity', 'N/A')
                                    }
                                }
                                
                                # Format percentages
                                for category, ratios in ratio_data.items():
                                    for key, value in ratios.items():
                                        if key in ["Profit Margin", "Operating Margin", "Return on Assets", 
                                                 "Return on Equity", "Dividend Yield", "Payout Ratio"]:
                                            if isinstance(value, (int, float)):
                                                ratio_data[category][key] = f"{value * 100:.2f}%"
                                
                                # Display ratios in expandable sections
                                for category, ratios in ratio_data.items():
                                    with st.expander(category, expanded=True):
                                        # Create a DataFrame for display
                                        df = pd.DataFrame.from_dict(ratios, orient='index', columns=['Value'])
                                        st.table(df)
                            
                            with fin_tabs[1]:
                                # Income Statement
                                st.markdown("### Income Statement")
                                
                                try:
                                    income_stmt = stock.income_stmt
                                    if not income_stmt.empty:
                                        st.dataframe(income_stmt)
                                    else:
                                        st.info("Income statement data not available")
                                except Exception as e:
                                    st.info(f"Could not retrieve income statement: {str(e)}")
                                    
                                # Key income metrics
                                st.markdown("### Key Income Metrics")
                                income_metrics = {
                                    "Total Revenue": info.get('totalRevenue', 'N/A'),
                                    "Revenue Per Share": info.get('revenuePerShare', 'N/A'),
                                    "Gross Profits": info.get('grossProfits', 'N/A'),
                                    "EBITDA": info.get('ebitda', 'N/A'),
                                    "Net Income": info.get('netIncomeToCommon', 'N/A'),
                                    "EPS (Trailing)": info.get('trailingEps', 'N/A'),
                                    "EPS (Forward)": info.get('forwardEps', 'N/A')
                                }
                                
                                # Format large numbers
                                for key, value in income_metrics.items():
                                    if isinstance(value, (int, float)) and key not in ["Revenue Per Share", "EPS (Trailing)", "EPS (Forward)"]:
                                        if abs(value) >= 1e9:
                                            income_metrics[key] = f"${value/1e9:.2f} B"
                                        elif abs(value) >= 1e6:
                                            income_metrics[key] = f"${value/1e6:.2f} M"
                                        else:
                                            income_metrics[key] = f"${value:,.2f}"
                                    elif isinstance(value, (int, float)):
                                        income_metrics[key] = f"${value:.2f}"
                                
                                df_income = pd.DataFrame.from_dict(income_metrics, orient='index', columns=['Value'])
                                st.table(df_income)
                            
                            with fin_tabs[2]:
                                # Balance Sheet
                                st.markdown("### Balance Sheet")
                                
                                try:
                                    balance_sheet = stock.balance_sheet
                                    if not balance_sheet.empty:
                                        st.dataframe(balance_sheet)
                                    else:
                                        st.info("Balance sheet data not available")
                                except Exception as e:
                                    st.info(f"Could not retrieve balance sheet: {str(e)}")
                                    
                                # Key balance sheet metrics
                                st.markdown("### Key Balance Sheet Metrics")
                                bs_metrics = {
                                    "Total Assets": info.get('totalAssets', 'N/A'),
                                    "Total Debt": info.get('totalDebt', 'N/A'),
                                    "Total Cash": info.get('totalCash', 'N/A'),
                                    "Total Equity": info.get('totalStockholderEquity', 'N/A'),
                                    "Book Value Per Share": info.get('bookValue', 'N/A'),
                                    "Cash Per Share": info.get('totalCashPerShare', 'N/A')
                                }
                                
                                # Format large numbers
                                for key, value in bs_metrics.items():
                                    if isinstance(value, (int, float)) and key not in ["Book Value Per Share", "Cash Per Share"]:
                                        if abs(value) >= 1e9:
                                            bs_metrics[key] = f"${value/1e9:.2f} B"
                                        elif abs(value) >= 1e6:
                                            bs_metrics[key] = f"${value/1e6:.2f} M"
                                        else:
                                            bs_metrics[key] = f"${value:,.2f}"
                                    elif isinstance(value, (int, float)):
                                        bs_metrics[key] = f"${value:.2f}"
                                
                                df_bs = pd.DataFrame.from_dict(bs_metrics, orient='index', columns=['Value'])
                                st.table(df_bs)
                            
                            with fin_tabs[3]:
                                # Cash Flow
                                st.markdown("### Cash Flow Statement")
                                
                                try:
                                    cashflow = stock.cashflow
                                    if not cashflow.empty:
                                        st.dataframe(cashflow)
                                    else:
                                        st.info("Cash flow data not available")
                                except Exception as e:
                                    st.info(f"Could not retrieve cash flow statement: {str(e)}")
                                    
                                # Key cash flow metrics
                                st.markdown("### Key Cash Flow Metrics")
                                cf_metrics = {
                                    "Operating Cash Flow": info.get('operatingCashflow', 'N/A'),
                                    "Free Cash Flow": info.get('freeCashflow', 'N/A'),
                                    "Capital Expenditures": info.get('capitalExpenditures', 'N/A'),
                                    "Dividend Paid": info.get('dividendsPaid', 'N/A')
                                }
                                
                                # Format large numbers
                                for key, value in cf_metrics.items():
                                    if isinstance(value, (int, float)):
                                        if abs(value) >= 1e9:
                                            cf_metrics[key] = f"${value/1e9:.2f} B"
                                        elif abs(value) >= 1e6:
                                            cf_metrics[key] = f"${value/1e6:.2f} M"
                                        else:
                                            cf_metrics[key] = f"${value:,.2f}"
                                
                                df_cf = pd.DataFrame.from_dict(cf_metrics, orient='index', columns=['Value'])
                                st.table(df_cf)
                    
                    except Exception as e:
                        st.error(f"Error analyzing stock: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())

# Add CSS for improved visuals
def local_css():
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    
    .metric-card h3, .metric-card h4 {
        margin-top: 0;
        margin-bottom: 10px;
        color: #333;
    }
    
    .metric-card p {
        margin-bottom: 5px;
        color: #555;
    }
    
    .recommendation-buy {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    .recommendation-hold {
        background-color: #ffc107;
        color: black;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    .recommendation-sell {
        background-color: #dc3545;
        color: white;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    .news-summary {
        border-left: 4px solid #007bff;
        padding-left: 15px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load stock data function
def load_stock_data():
    # This is a simplified function that would normally load data from a file
    # Here we just return a small dict of example stocks
    stock_dict = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com, Inc.",
        "META": "Meta Platforms, Inc.",
        "TSLA": "Tesla, Inc.",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc.",
        "RELIANCE.NS": "Reliance Industries Limited",
        "TATAMOTORS.NS": "Tata Motors Limited",
        "HDFCBANK.NS": "HDFC Bank Limited",
        "TCS.NS": "Tata Consultancy Services Limited"
    }
    
    # Create a reverse mapping
    name_to_ticker = {v.lower(): k for k, v in stock_dict.items()}
    
    return stock_dict, name_to_ticker

# Calculate RSI function
def calculate_rsi(prices, period=14):
    # Calculate RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
# Calculate MACD function
def calculate_macd(prices, fast=12, slow=26, signal=9):
    # Calculate MACD
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line.iloc[-1], signal_line.iloc[-1]

# Determine trend function
def determine_trend(hist):
    if len(hist) < 20:
        return {
            "trend": "Insufficient data for trend analysis",
            "1d_change": "N/A",
            "1w_change": "N/A",
            "1m_change": "N/A"
        }
    
    # Get closing prices
    close = hist['Close']
    
    # Calculate moving averages
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()
    ma200 = close.rolling(window=200).mean()
    
    # Calculate daily, weekly, and monthly changes
    daily_change = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100
    weekly_change = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100 if len(close) >= 5 else 0
    monthly_change = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100 if len(close) >= 20 else 0
    
    # Determine trend based on moving averages
    current_price = close.iloc[-1]
    trend = ""
    
    if current_price > ma20.iloc[-1] and current_price > ma50.iloc[-1] and current_price > ma200.iloc[-1]:
        if ma20.iloc[-1] > ma50.iloc[-1] and ma50.iloc[-1] > ma200.iloc[-1]:
            trend = "Strong Uptrend"
        else:
            trend = "Uptrend"
    elif current_price < ma20.iloc[-1] and current_price < ma50.iloc[-1] and current_price < ma200.iloc[-1]:
        if ma20.iloc[-1] < ma50.iloc[-1] and ma50.iloc[-1] < ma200.iloc[-1]:
            trend = "Strong Downtrend"
        else:
            trend = "Downtrend"
    elif current_price > ma200.iloc[-1]:
        if current_price > ma50.iloc[-1]:
            trend = "Bullish"
        else:
            trend = "Neutral with Bullish Bias"
    elif current_price < ma200.iloc[-1]:
        if current_price < ma50.iloc[-1]:
            trend = "Bearish"
        else:
            trend = "Neutral with Bearish Bias"
    else:
        trend = "Neutral"
    
    # Format change percentages
    daily_change_str = f"{daily_change:+.2f}%"
    weekly_change_str = f"{weekly_change:+.2f}%"
    monthly_change_str = f"{monthly_change:+.2f}%"
    
    return {
        "trend": trend,
        "1d_change": daily_change_str,
        "1w_change": weekly_change_str,
        "1m_change": monthly_change_str
    }

# Get news function
def get_comprehensive_news(ticker):
    try:
        # In a real implementation, this would call an API to get news
        # For this example, we'll return sample data
        return [
            {
                "title": f"Analysis: {ticker} Reports Strong Quarterly Results",
                "publisher": "Market News",
                "publishedDate": "2025-04-10",
                "summary": f"{ticker} reported earnings that beat analyst expectations, driven by strong product demand.",
                "link": "#"
            },
            {
                "title": f"Investors Positive on {ticker}'s Future Outlook",
                "publisher": "Investment Daily",
                "publishedDate": "2025-04-08",
                "summary": f"Analysts see potential for {ticker} to maintain growth momentum through the rest of 2025.",
                "link": "#"
            },
            {
                "title": f"{ticker} Announces New Product Line",
                "publisher": "Tech Report",
                "publishedDate": "2025-04-05",
                "summary": f"{ticker} unveiled a new product line scheduled for release in Q3 2025.",
                "link": "#"
            }
        ]
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

# Summarize news function
def summarize_news(news_items):
    if not news_items:
        return "No recent news available for analysis."
    
    # In a real implementation, this might use NLP or a more sophisticated approach
    # For this example, we'll just provide a simple summary
    
    # Generate random sentiment for demo purposes
    import random
    sentiments = ["very positive", "moderately positive", "neutral", "moderately negative", "very negative"]
    sentiment = random.choice(sentiments)
    
    return f"""
    Based on recent news articles, the company has been involved in product announcements and financial reporting.
    News Sentiment: {sentiment}. The most recent news focuses on quarterly results and future outlook.
    """

# Initialize session state for data caching
if 'stock_data_cache' not in st.session_state:
    st.session_state.stock_data_cache = {}

# Call CSS function
local_css()
# Run the main app
if __name__ == "__main__":
    main()
                                            
                                            
