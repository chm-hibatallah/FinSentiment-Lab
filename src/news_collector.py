"""
Main Financial News Collector
Single source of truth for collecting financial news
Collects maximum articles from NewsAPI and RSS feeds
"""

import requests
import feedparser
import pandas as pd
import time
from datetime import datetime
import os
import shutil
from src.config import (
    NEWS_API_KEY,
    OUTPUT_FILE,
    RSS_FEEDS,
    NEWSAPI_QUERIES,
    MAX_ARTICLES_TOTAL,
    MAX_ARTICLES_PER_NEWSAPI_QUERY,
    MAX_ARTICLES_PER_RSS_FEED,
    REQUEST_DELAY,
    RSS_DELAY,
    REQUEST_TIMEOUT,
    USE_TIMESTAMP_IN_FILENAME,
    BACKUP_OLD_DATA
)

def collect_from_newsapi():
    """
    Collect articles from NewsAPI using multiple queries
    Returns list of raw article dictionaries
    """
    print("\nCollecting from NewsAPI...")
    print("-" * 40)
    
    all_articles = []
    total_queries = len(NEWSAPI_QUERIES)
    
    for i, query in enumerate(NEWSAPI_QUERIES, 1):
        print(f"  Query {i}/{total_queries}: '{query}'", end="")
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": MAX_ARTICLES_PER_NEWSAPI_QUERY,
            "apiKey": NEWS_API_KEY
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                
                for item in articles:
                    article = {
                        "title": item.get("title", ""),
                        "published": item.get("publishedAt", ""),
                        "source": item.get("source", {}).get("name", "Unknown"),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "description": item.get("description", ""),
                        "author": item.get("author", ""),
                        "collection_method": "newsapi",
                        "collection_query": query,
                        "collected_at": datetime.now().isoformat()
                    }
                    all_articles.append(article)
                
                print(f" - {len(articles)} articles")
            else:
                print(f" - Error: {data.get('message', 'Unknown')}")
                
        except Exception as e:
            print(f" - Error: {str(e)}")
        
        time.sleep(REQUEST_DELAY)
    
    print(f"\nTotal NewsAPI articles: {len(all_articles)}")
    return all_articles

def collect_from_rss():
    """
    Collect articles from RSS feeds
    Returns list of raw article dictionaries
    """
    print("\nCollecting from RSS Feeds...")
    print("-" * 40)
    
    all_articles = []
    total_feeds = len(RSS_FEEDS)
    
    for i, feed in enumerate(RSS_FEEDS, 1):
        print(f"  Feed {i}/{total_feeds}: {feed['name']}", end="")
        
        try:
            parsed = feedparser.parse(feed["url"])
            
            feed_articles = []
            for entry in parsed.entries[:MAX_ARTICLES_PER_RSS_FEED]:
                
                # Get content (could be in different fields)
                content = ""
                if hasattr(entry, "content") and entry.content:
                    content = entry.content[0].value
                elif hasattr(entry, "summary"):
                    content = entry.summary
                elif hasattr(entry, "description"):
                    content = entry.description
                
                article = {
                    "title": entry.get("title", ""),
                    "published": entry.get("published", entry.get("updated", "")),
                    "source": feed["name"],
                    "url": entry.get("link", ""),
                    "content": content,
                    "description": entry.get("summary", ""),
                    "author": entry.get("author", ""),
                    "collection_method": "rss",
                    "collection_feed": feed["name"],
                    "collected_at": datetime.now().isoformat()
                }
                feed_articles.append(article)
            
            all_articles.extend(feed_articles)
            print(f" - {len(feed_articles)} articles")
            
        except Exception as e:
            print(f" - Error: {str(e)}")
        
        time.sleep(RSS_DELAY)
    
    print(f"\nTotal RSS articles: {len(all_articles)}")
    return all_articles

def remove_duplicates(articles):
    """
    Simple duplicate removal based on URL
    """
    if not articles:
        return articles
    
    print("\nRemoving duplicates...")
    
    df = pd.DataFrame(articles)
    before = len(df)
    
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    
    after = len(df)
    print(f"  Removed {before - after} duplicate articles")
    print(f"  Final unique articles: {after}")
    
    return df.to_dict('records')

def save_raw_data(articles, filename):
    """
    Save raw articles to CSV file
    No preprocessing - just raw data
    """
    if not articles:
        print("\nNo articles to save")
        return False
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False)
    
    print(f"\nData saved to: {filename}")
    print(f"Total articles in file: {len(df)}")
    
    return True

def show_sample(articles, n=3):
    """
    Show sample of collected articles
    """
    if not articles:
        return
    
    print("\nSample Articles:")
    print("-" * 60)
    
    for i, article in enumerate(articles[:n]):
        print(f"\nArticle {i+1}:")
        print(f"  Title: {article.get('title', 'N/A')[:100]}")
        print(f"  Source: {article.get('source', 'N/A')}")
        print(f"  Published: {article.get('published', 'N/A')}")
        print(f"  Method: {article.get('collection_method', 'N/A')}")
        
        content = article.get('content', '')
        if content:
            preview = content[:150].replace('\n', ' ').replace('\r', '')
            print(f"  Preview: {preview}...")

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """
    Main function to run the raw data collection
    """
    print("=" * 60)
    print("RAW FINANCIAL NEWS COLLECTOR")
    print("=" * 60)
    print("This script collects raw data without any preprocessing")
    
    # Step 1: Collect from NewsAPI
    newsapi_articles = collect_from_newsapi()
    
    # Step 2: Collect from RSS
    rss_articles = collect_from_rss()
    
    # Step 3: Combine all articles
    all_articles = newsapi_articles + rss_articles
    print(f"\nTotal articles collected: {len(all_articles)}")
    
    # Step 4: Remove duplicates
    if all_articles:
        all_articles = remove_duplicates(all_articles)
    
    # Step 5: Save raw data
    if all_articles:
        save_raw_data(all_articles, OUTPUT_FILE)
        
        # Show sample
        show_sample(all_articles, 3)
        
        # Print summary
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total unique articles: {len(all_articles)}")
        print(f"NewsAPI articles: {len(newsapi_articles)}")
        print(f"RSS articles: {len(rss_articles)}")
        
        # Count by source
        df = pd.DataFrame(all_articles)
        if "source" in df.columns:
            top_sources = df["source"].value_counts().head(5)
            print("\nTop 5 Sources:")
            for source, count in top_sources.items():
                print(f"  {source}: {count}")
    else:
        print("\nNo articles were collected. Check your internet connection and API key.")

# ============================================
# RUN THE SCRIPT
# ============================================

if __name__ == "__main__":
    main()
    print("\nDone!")