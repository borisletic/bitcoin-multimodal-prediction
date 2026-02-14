"""
Preprocessing Pipeline for Bitcoin Multimodal Prediction
Author: Boris Letić (Student 1 - Sentiment Analysis)
Date: 2025-02-13

This script handles preprocessing of:
1. Twitter sentiment data (cleaning, tokenization, labeling)
2. Feature engineering for sentiment (rolling aggregates, engagement metrics)
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse
import os
from datetime import datetime
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TweetPreprocessor:
    """
    Preprocessor for Twitter sentiment data
    Handles cleaning, tokenization, and feature engineering
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep some important crypto-related words that might be in stopwords
        crypto_keywords = {'not', 'no', 'but', 'more', 'down', 'up', 'above', 'below'}
        self.stop_words = self.stop_words - crypto_keywords
        
        print(" Tweet Preprocessor initialized")
    
    
    def clean_tweet(self, text):
        """
        Clean a single tweet text
        
        Steps:
        1. Remove URLs
        2. Remove mentions (@username)
        3. Remove hashtags (#)
        4. Remove special characters
        5. Convert to lowercase
        6. Remove extra whitespace
        
        Args:
            text: Raw tweet text
        
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbol but keep the word
        text = re.sub(r'#', '', text)
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text
        
        Args:
            text: Cleaned text
        
        Returns:
            List of tokens
        """
        if not text or text == "":
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    
    def process_tweets(self, df, text_column='text'):
        """
        Process all tweets in a DataFrame
        
        Args:
            df: DataFrame with tweets
            text_column: Name of column containing tweet text
        
        Returns:
            DataFrame with additional columns:
                - cleaned_text: Cleaned tweet text
                - tokens: List of tokens
                - token_count: Number of tokens
        """
        print(f"\n{'='*60}")
        print(f"Processing {len(df)} tweets...")
        print(f"{'='*60}")
        
        # Apply cleaning
        print("\n[1/3] Cleaning tweets...")
        tqdm.pandas(desc="Cleaning")
        df['cleaned_text'] = df[text_column].progress_apply(self.clean_tweet)
        
        # Apply tokenization
        print("\n[2/3] Tokenizing and lemmatizing...")
        tqdm.pandas(desc="Tokenizing")
        df['tokens'] = df['cleaned_text'].progress_apply(self.tokenize_and_lemmatize)
        
        # Count tokens
        print("\n[3/3] Counting tokens...")
        df['token_count'] = df['tokens'].apply(len)
        
        # Filter out empty tweets
        original_len = len(df)
        df = df[df['token_count'] > 0].copy()
        filtered_count = original_len - len(df)
        
        print(f"\n Preprocessing complete!")
        print(f"  - Original tweets: {original_len}")
        print(f"  - Filtered (empty): {filtered_count}")
        print(f"  - Remaining tweets: {len(df)}")
        print(f"  - Avg tokens per tweet: {df['token_count'].mean():.2f}")
        
        return df


def label_sentiment_manual(df, positive_keywords, negative_keywords):
    """
    Manual sentiment labeling using keyword matching
    This is a simple heuristic for initial labeling
    
    Args:
        df: DataFrame with cleaned tweets
        positive_keywords: List of positive keywords
        negative_keywords: List of negative keywords
    
    Returns:
        DataFrame with sentiment labels
    """
    print(f"\n{'='*60}")
    print(f"Labeling sentiment using keyword matching...")
    print(f"{'='*60}")
    
    def classify_sentiment(tokens):
        if not tokens:
            return 'neutral'
        
        positive_count = sum(1 for token in tokens if token in positive_keywords)
        negative_count = sum(1 for token in tokens if token in negative_keywords)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    df['sentiment'] = df['tokens'].apply(classify_sentiment)
    
    # Print distribution
    sentiment_counts = df['sentiment'].value_counts()
    print(f"\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    return df


def calculate_engagement_score(row):
    """
    Calculate weighted engagement score from likes and retweets
    
    Args:
        row: DataFrame row with 'likes' and 'retweets' columns
    
    Returns:
        Engagement score (float)
    """
    # Weighted formula: retweets are more valuable than likes
    likes = row.get('likes', 0) if pd.notna(row.get('likes', 0)) else 0
    retweets = row.get('retweets', 0) if pd.notna(row.get('retweets', 0)) else 0
    
    engagement = (likes * 1.0) + (retweets * 2.0)
    return engagement


def engineer_sentiment_features(df):
    """
    Engineer additional features from sentiment data
    
    Creates:
    - Engagement score (weighted likes + retweets)
    - Hour of day, day of week
    - Rolling sentiment aggregates (6h, 12h, 24h windows)
    
    Args:
        df: DataFrame with sentiment labels
    
    Returns:
        DataFrame with additional features
    """
    print(f"\n{'='*60}")
    print(f"Engineering sentiment features...")
    print(f"{'='*60}")
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 1. Engagement score
    print("\n[1/5] Calculating engagement scores...")
    df['engagement_score'] = df.apply(calculate_engagement_score, axis=1)
    
    # 2. Temporal features
    print("[2/5] Extracting temporal features...")
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # 3. Sentiment encoding for numerical aggregation
    print("[3/5] Encoding sentiment...")
    sentiment_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    
    # 4. Hourly aggregation
    print("[4/5] Aggregating to hourly intervals...")
    df['datetime_hour'] = df['datetime'].dt.floor('h')
    
    hourly_agg = df.groupby('datetime_hour').agg({
        'sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
        'engagement_score': ['sum', 'mean', 'max'],
        'token_count': 'mean'
    }).reset_index()
    
    # Flatten column names
    hourly_agg.columns = ['_'.join(col).strip('_') for col in hourly_agg.columns.values]
    hourly_agg.rename(columns={'datetime_hour': 'datetime'}, inplace=True)
    
    # 5. Rolling window aggregates
    print("[5/5] Computing rolling window aggregates...")
    
    # Sort by datetime
    hourly_agg = hourly_agg.sort_values('datetime')
    
    # Define windows (in hours)
    windows = [6, 12, 24]
    
    for window in windows:
        # Sentiment rolling mean
        hourly_agg[f'sentiment_mean_{window}h'] = (
            hourly_agg['sentiment_score_mean']
            .rolling(window=window, min_periods=1)
            .mean()
        )
        
        # Sentiment rolling std
        hourly_agg[f'sentiment_std_{window}h'] = (
            hourly_agg['sentiment_score_std']
            .rolling(window=window, min_periods=1)
            .mean()
        )
        
        # Tweet volume
        hourly_agg[f'tweet_volume_{window}h'] = (
            hourly_agg['sentiment_score_count']
            .rolling(window=window, min_periods=1)
            .sum()
        )
        
        # Engagement
        hourly_agg[f'engagement_{window}h'] = (
            hourly_agg['engagement_score_sum']
            .rolling(window=window, min_periods=1)
            .sum()
        )
    
    print(f"\n Feature engineering complete!")
    print(f"  - Total features created: {len(hourly_agg.columns)}")
    print(f"  - Hourly data points: {len(hourly_agg)}")
    
    return hourly_agg


def split_data_temporal(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split data temporally (no shuffle) for time series
    
    Args:
        df: DataFrame sorted by datetime
        train_ratio: Proportion for training (default 0.6)
        val_ratio: Proportion for validation (default 0.2)
        test_ratio: Proportion for testing (default 0.2)
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\n{'='*60}")
    print(f"Data Split (Temporal)")
    print(f"{'='*60}")
    print(f"Training:   {len(train_df):5d} samples ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"Validation: {len(val_df):5d} samples ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    print(f"Test:       {len(test_df):5d} samples ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    print(f"Total:      {n:5d} samples")
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Preprocess Twitter sentiment data')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with raw tweets')
    parser.add_argument('--output_dir', type=str, default='data/processed/',
                       help='Output directory for processed data')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of column containing tweet text')
    parser.add_argument('--datetime_column', type=str, default='datetime',
                       help='Name of column containing datetime')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# Twitter Sentiment Data Preprocessing")
    print(f"# Author: Boris Letic (Student 1)")
    print(f"{'#'*60}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {args.input}")
    df = pd.read_csv(args.input)
    
    # Convert datetime column
    if args.datetime_column in df.columns:
        df['datetime'] = pd.to_datetime(df[args.datetime_column])
    else:
        print(f"Warning: '{args.datetime_column}' column not found. Using current time.")
        df['datetime'] = pd.Timestamp.now()
    
    print(f" Loaded {len(df)} tweets")
    
    # Initialize preprocessor
    preprocessor = TweetPreprocessor()
    
    # Process tweets
    df = preprocessor.process_tweets(df, text_column=args.text_column)
    
    # Label sentiment (simple keyword-based for now)
    # TODO: Replace with BERT fine-tuned model later
    positive_keywords = {'bullish', 'moon', 'pump', 'buy', 'hodl', 'gains', 'profit', 
                        'rise', 'surge', 'rally', 'good', 'great', 'awesome', 'best'}
    negative_keywords = {'bearish', 'dump', 'sell', 'crash', 'drop', 'fall', 'loss',
                        'fear', 'panic', 'bad', 'worst', 'scam', 'fraud'}
    
    df = label_sentiment_manual(df, positive_keywords, negative_keywords)
    
    # Save individual tweets with sentiment labels
    tweets_output = os.path.join(args.output_dir, 'tweets_labeled.csv')
    df.to_csv(tweets_output, index=False)
    print(f"\n Labeled tweets saved to: {tweets_output}")
    
    # Engineer features
    features_df = engineer_sentiment_features(df)
    
    # Split data
    train_df, val_df, test_df = split_data_temporal(features_df)
    
    # Save splits
    train_path = os.path.join(args.output_dir, 'sentiment_features_train.csv')
    val_path = os.path.join(args.output_dir, 'sentiment_features_val.csv')
    test_path = os.path.join(args.output_dir, 'sentiment_features_test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n Training data saved to: {train_path}")
    print(f" Validation data saved to: {val_path}")
    print(f" Test data saved to: {test_path}")
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
