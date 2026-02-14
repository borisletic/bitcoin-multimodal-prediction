"""
Data Collection Script for Bitcoin Multimodal Prediction
Author: Boris Letić & Bogdan Čiplić
Date: 2025-02-13

This script handles data collection from multiple sources:
1. Twitter sentiment data (Kaggle or Twitter API)
2. Bitcoin price data (Yahoo Finance)
"""

import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

def collect_bitcoin_prices(ticker='BTC-USD', start_date='2024-01-01', end_date='2024-05-31', interval='1h'):
    """
    Collect Bitcoin price data from Yahoo Finance
    
    Args:
        ticker: Trading pair ticker (default: BTC-USD)
        start_date: Start date for data collection (YYYY-MM-DD)
        end_date: End date for data collection (YYYY-MM-DD)
        interval: Data interval (1h, 1d, etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n{'='*60}")
    print(f"Collecting Bitcoin Price Data")
    print(f"{'='*60}")
    print(f"Ticker: {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    
    try:
        # Download data
        btc = yf.Ticker(ticker)
        df = btc.history(start=start_date, end=end_date, interval=interval)
        
        # Reset index to make datetime a column
        df.reset_index(inplace=True)
        
        # Rename columns for consistency
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        print(f"\n✓ Successfully collected {len(df)} data points")
        print(f"✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
    
    except Exception as e:
        print(f"\n✗ Error collecting price data: {e}")
        sys.exit(1)


def collect_twitter_data_kaggle(kaggle_dataset_path):
    """
    Load Twitter sentiment data from Kaggle dataset
    
    Args:
        kaggle_dataset_path: Path to Kaggle CSV file
    
    Returns:
        DataFrame with tweets
    """
    print(f"\n{'='*60}")
    print(f"Loading Twitter Data from Kaggle")
    print(f"{'='*60}")
    print(f"Dataset path: {kaggle_dataset_path}")
    
    try:
        # Load CSV
        df = pd.read_csv(kaggle_dataset_path)
        
        print(f"\n✓ Successfully loaded {len(df)} tweets")
        print(f"\nDataset columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
    
    except FileNotFoundError:
        print(f"\n✗ Error: File not found at {kaggle_dataset_path}")
        print("\nPlease download the Kaggle 'Bitcoin Tweets' dataset:")
        print("https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error loading Twitter data: {e}")
        sys.exit(1)


def collect_twitter_data_api(api_keys, search_query='#Bitcoin OR #BTC OR $BTC', 
                             start_date='2024-01-01', end_date='2024-05-31', max_tweets=50000):
    """
    Collect Twitter data using Twitter API (requires API keys)
    
    Args:
        api_keys: Dictionary with Twitter API credentials
        search_query: Search query for tweets
        start_date: Start date for collection
        end_date: End date for collection
        max_tweets: Maximum number of tweets to collect
    
    Returns:
        DataFrame with tweets
    """
    print(f"\n{'='*60}")
    print(f"Collecting Twitter Data via API")
    print(f"{'='*60}")
    
    try:
        import tweepy
        
        # Authenticate with Twitter API
        auth = tweepy.OAuthHandler(api_keys['consumer_key'], api_keys['consumer_secret'])
        auth.set_access_token(api_keys['access_token'], api_keys['access_token_secret'])
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        print(f"✓ Authentication successful")
        print(f"Query: {search_query}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Max tweets: {max_tweets}")
        
        # Collect tweets
        tweets_data = []
        
        for tweet in tweepy.Cursor(api.search_tweets,
                                   q=search_query,
                                   lang="en",
                                   since=start_date,
                                   until=end_date,
                                   tweet_mode='extended').items(max_tweets):
            
            tweets_data.append({
                'datetime': tweet.created_at,
                'text': tweet.full_text,
                'user': tweet.user.screen_name,
                'likes': tweet.favorite_count,
                'retweets': tweet.retweet_count,
                'tweet_id': tweet.id
            })
            
            if len(tweets_data) % 1000 == 0:
                print(f"Collected {len(tweets_data)} tweets...")
        
        df = pd.DataFrame(tweets_data)
        print(f"\n✓ Successfully collected {len(df)} tweets")
        
        return df
    
    except ImportError:
        print("\n✗ Error: tweepy library not installed")
        print("Install with: pip install tweepy")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error collecting Twitter data: {e}")
        sys.exit(1)


def save_data(df, output_path, data_type='prices'):
    """
    Save collected data to CSV
    
    Args:
        df: DataFrame to save
        output_path: Output directory
        data_type: Type of data ('prices' or 'tweets')
    """
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{data_type}_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    df.to_csv(filepath, index=False)
    print(f"\n✓ Data saved to: {filepath}")
    print(f"✓ File size: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Bitcoin Multimodal Data Collection')
    
    # Source selection
    parser.add_argument('--source', type=str, choices=['yfinance', 'kaggle', 'twitter_api'],
                       required=True, help='Data source')
    
    # Common arguments
    parser.add_argument('--output', type=str, default='data/',
                       help='Output directory for collected data')
    
    # Bitcoin price arguments
    parser.add_argument('--ticker', type=str, default='BTC-USD',
                       help='Trading pair ticker (for yfinance)')
    parser.add_argument('--interval', type=str, default='1h',
                       help='Data interval (1h, 1d, etc.)')
    
    # Date range arguments
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-05-31',
                       help='End date (YYYY-MM-DD)')
    
    # Twitter-specific arguments
    parser.add_argument('--kaggle_path', type=str,
                       help='Path to Kaggle dataset CSV (for kaggle source)')
    parser.add_argument('--max_tweets', type=int, default=50000,
                       help='Maximum tweets to collect (for twitter_api source)')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# Bitcoin Multimodal Data Collection")
    print(f"# Authors: Boris Letić & Bogdan Čiplić")
    print(f"{'#'*60}")
    
    # Collect data based on source
    if args.source == 'yfinance':
        df = collect_bitcoin_prices(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval
        )
        save_data(df, args.output, data_type='btc_prices')
    
    elif args.source == 'kaggle':
        if not args.kaggle_path:
            print("\n✗ Error: --kaggle_path required for Kaggle source")
            sys.exit(1)
        
        df = collect_twitter_data_kaggle(args.kaggle_path)
        save_data(df, args.output, data_type='tweets')
    
    elif args.source == 'twitter_api':
        print("\n⚠ Warning: Twitter API requires authentication keys")
        print("Please provide API keys as environment variables:")
        print("  - TWITTER_CONSUMER_KEY")
        print("  - TWITTER_CONSUMER_SECRET")
        print("  - TWITTER_ACCESS_TOKEN")
        print("  - TWITTER_ACCESS_TOKEN_SECRET")
        
        # Check for API keys in environment
        api_keys = {
            'consumer_key': os.getenv('TWITTER_CONSUMER_KEY'),
            'consumer_secret': os.getenv('TWITTER_CONSUMER_SECRET'),
            'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
            'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        }
        
        if not all(api_keys.values()):
            print("\n✗ Error: Missing API keys in environment variables")
            sys.exit(1)
        
        df = collect_twitter_data_api(
            api_keys=api_keys,
            start_date=args.start,
            end_date=args.end,
            max_tweets=args.max_tweets
        )
        save_data(df, args.output, data_type='tweets')
    
    print(f"\n{'='*60}")
    print(f"Data Collection Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
