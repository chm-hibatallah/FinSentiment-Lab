"""
Financial News Preprocessing Script
Processes raw CSV files from the collector and prepares them for analysis
"""

import pandas as pd
import re
import os
from datetime import datetime
from src.config import PROCESSED_DATA_PATH, MIN_WORDS_FOR_ANALYSIS

def clean_article_text(text):
    """
    Clean and normalize article text
    
    Args:
        text: Raw article text
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    
    # Remove special characters (keep letters, numbers, spaces, and basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-\$\%]", " ", text)
    
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    return text

def count_words(text):
    """
    Count number of words in text
    
    Args:
        text: Input text
    
    Returns:
        Word count
    """
    if not isinstance(text, str):
        return 0
    
    words = text.split()
    return len(words)

def count_sentences(text):
    """
    Count number of sentences in text
    
    Args:
        text: Input text
    
    Returns:
        Sentence count
    """
    if not isinstance(text, str):
        return 0
    
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

def extract_financial_terms(text):
    """
    Count financial terms in text
    
    Args:
        text: Input text
    
    Returns:
        Count of financial terms
    """
    if not isinstance(text, str):
        return 0
    
    financial_terms = [
        "stock", "stocks", "market", "markets", "share", "shares",
        "invest", "investing", "investment", "investor", "investors",
        "trading", "trade", "trader", "traders",
        "bond", "bonds", "etf", "etfs", "fund", "funds",
        "dividend", "dividends", "earnings", "revenue", "profit", "loss",
        "ceo", "cfo", "company", "companies", "corporation",
        "bank", "banks", "banking", "financial", "finance",
        "economy", "economic", "fed", "federal", "reserve",
        "interest", "rate", "rates", "inflation", "deflation",
        "bull", "bear", "rally", "crash", "correction", "volatility",
        "ipo", "merger", "acquisition", "takeover", "buyout",
        "quarter", "quarterly", "annual", "fiscal", "earnings"
    ]
    
    text_lower = text.lower()
    count = 0
    
    for term in financial_terms:
        count += text_lower.count(term)
    
    return count

def count_money_mentions(text):
    """
    Count mentions of money amounts
    
    Args:
        text: Input text
    
    Returns:
        Count of money mentions
    """
    if not isinstance(text, str):
        return 0
    
    patterns = [
        r"\$\d+(?:\.\d+)?",  # $100, $100.50
        r"\d+\s?(?:dollars|usd|bucks)",  # 100 dollars
        r"\$\d+\s?(?:million|billion|trillion|m|b|t)",  # $1 million, $1B
        r"(?:million|billion|trillion)\s?dollars"  # million dollars
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        count += len(matches)
    
    return count

def count_percentages(text):
    """
    Count mentions of percentages
    
    Args:
        text: Input text
    
    Returns:
        Count of percentage mentions
    """
    if not isinstance(text, str):
        return 0
    
    patterns = [
        r"\d+(?:\.\d+)?%",  # 10%, 10.5%
        r"\d+\s?percent",  # 10 percent
        r"percentage\s?points?"  # percentage point
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        count += len(matches)
    
    return count

def extract_date_features(date_string):
    """
    Extract features from date
    
    Args:
        date_string: Date string
    
    Returns:
        Dictionary with date features
    """
    features = {
        "year": None,
        "month": None,
        "day": None,
        "day_of_week": None,
        "hour": None
    }
    
    if pd.isna(date_string) or not isinstance(date_string, str):
        return features
    
    try:
        date_obj = pd.to_datetime(date_string, errors="coerce")
        
        if pd.notna(date_obj):
            features["year"] = date_obj.year
            features["month"] = date_obj.month
            features["day"] = date_obj.day
            features["day_of_week"] = date_obj.day_name()
            features["hour"] = date_obj.hour
    except:
        pass
    
    return features

def process_raw_file(input_file, output_file=None):
    """
    Process a raw CSV file from the collector
    
    Args:
        input_file: Path to input CSV file (from collector)
        output_file: Path to output processed CSV file
    
    Returns:
        Processed dataframe
    """
    print(f"\nProcessing file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"  Error: File not found - {input_file}")
        return None
    
    try:
        df = pd.read_csv(input_file)
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  Error loading file: {e}")
        return None
    
    print("  Cleaning and processing text...")
    
    processed_count = 0
    
    for index, row in df.iterrows():
        # Get text from either 'content' or 'description' column
        text = ""
        if "content" in df.columns and pd.notna(row.get("content")):
            text = str(row["content"])
        elif "description" in df.columns and pd.notna(row.get("description")):
            text = str(row["description"])
        elif "title" in df.columns and pd.notna(row.get("title")):
            text = str(row["title"])
        
        if not text:
            continue
        
        cleaned_text = clean_article_text(text)
        
        df.loc[index, "cleaned_text"] = cleaned_text
        df.loc[index, "word_count"] = count_words(cleaned_text)
        df.loc[index, "sentence_count"] = count_sentences(cleaned_text)
        
        word_count = df.loc[index, "word_count"]
        if word_count > 0:
            df.loc[index, "avg_word_length"] = len(cleaned_text.replace(" ", "")) / word_count
        else:
            df.loc[index, "avg_word_length"] = 0
            
        df.loc[index, "financial_term_count"] = extract_financial_terms(cleaned_text)
        df.loc[index, "money_mentions"] = count_money_mentions(cleaned_text)
        df.loc[index, "percentage_mentions"] = count_percentages(cleaned_text)
        df.loc[index, "has_financial_content"] = df.loc[index, "financial_term_count"] > 0
        
        processed_count += 1
        
        if index % 100 == 0 and index > 0:
            print(f"    Processed {index} rows...")
    
    print(f"  Processed {processed_count} rows with text")
    
    # Extract date features if published column exists
    if "published" in df.columns:
        print("  Extracting date features...")
        
        date_features = df["published"].apply(extract_date_features)
        
        df["year"] = date_features.apply(lambda x: x["year"])
        df["month"] = date_features.apply(lambda x: x["month"])
        df["day"] = date_features.apply(lambda x: x["day"])
        df["day_of_week"] = date_features.apply(lambda x: x["day_of_week"])
        df["hour"] = date_features.apply(lambda x: x["hour"])
    
    # Add processing metadata
    df["processing_date"] = datetime.now().strftime("%Y-%m-%d")
    
    # Calculate data quality score (adjusted for your data)
    max_word_count = df["word_count"].max() if len(df) > 0 else 1
    max_financial_terms = df["financial_term_count"].max() if len(df) > 0 else 1
    max_money = df["money_mentions"].max() if len(df) > 0 else 1
    max_percentage = df["percentage_mentions"].max() if len(df) > 0 else 1
    
    df["data_quality_score"] = (
        (df["word_count"] / max_word_count) * 0.4 +
        (df["financial_term_count"] / max_financial_terms) * 0.3 +
        (df["money_mentions"] / max_money) * 0.15 +
        (df["percentage_mentions"] / max_percentage) * 0.15
    )
    df["data_quality_score"] = df["data_quality_score"].fillna(0).clip(0, 1)
    
    # Flag articles suitable for analysis
    df["ready_for_analysis"] = (
        (df["word_count"] >= MIN_WORDS_FOR_ANALYSIS) & 
        (df["has_financial_content"] == True)
    )
    
    # Save processed file
    if output_file is None:
        # Create output filename based on input
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join("data/processed", f"processed_{name_without_ext}.csv")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        df.to_csv(output_file, index=False)
        print(f"  Saved {len(df)} rows to: {output_file}")
    except Exception as e:
        print(f"  Error saving file: {e}")
    
    return df

def find_raw_csv_files():
    """
    Find all raw CSV files in the data/raw directory
    
    Returns:
        List of CSV file paths
    """
    raw_dir = "data/raw"
    csv_files = []
    
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist")
        return csv_files
    
    for file in os.listdir(raw_dir):
        if file.endswith(".csv") and not file.startswith("processed_"):
            csv_files.append(os.path.join(raw_dir, file))
    
    return csv_files

def main():
    """
    Main function to process all raw news files
    """
    print("=" * 60)
    print("FINANCIAL NEWS PREPROCESSING PIPELINE")
    print("=" * 60)
    print("This script processes raw CSV files from the collector")
    
    # Find all raw CSV files
    csv_files = find_raw_csv_files()
    
    if not csv_files:
        print("\nNo raw CSV files found in data/raw/")
        print("Please run the collector first to get data")
        return
    
    print(f"\nFound {len(csv_files)} raw CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    all_processed_dfs = []
    
    # Process each file
    for file_path in csv_files:
        df = process_raw_file(file_path)
        
        if df is not None:
            df["original_file"] = os.path.basename(file_path)
            all_processed_dfs.append(df)
    
    # Combine all processed data if there are multiple files
    if len(all_processed_dfs) > 1:
        print("\n" + "=" * 60)
        print("COMBINING PROCESSED FILES")
        print("=" * 60)
        
        combined_df = pd.concat(all_processed_dfs, ignore_index=True)
        
        # Remove duplicates across files
        if "url" in combined_df.columns:
            before = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["url"], keep="first")
            after = len(combined_df)
            print(f"Removed {before - after} duplicate articles across files")
        
        # Save combined dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = f"data/processed/combined_news_{timestamp}.csv"
        combined_df.to_csv(combined_path, index=False)
        
        print(f"Saved combined dataset: {combined_path}")
        print(f"Total articles in combined dataset: {len(combined_df)}")
        
        # Also save to the main processed data path from config
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        combined_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Saved to main processed path: {PROCESSED_DATA_PATH}")
    
    elif len(all_processed_dfs) == 1:
        # Just one file, save to main path
        df = all_processed_dfs[0]
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"\nSaved to main processed path: {PROCESSED_DATA_PATH}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    
    if all_processed_dfs:
        # Load the final processed data
        if os.path.exists(PROCESSED_DATA_PATH):
            final_df = pd.read_csv(PROCESSED_DATA_PATH)
            
            print(f"\nFinal processed dataset: {len(final_df)} articles")
            
            if "collection_method" in final_df.columns:
                print("\nArticles by collection method:")
                method_counts = final_df["collection_method"].value_counts()
                for method, count in method_counts.items():
                    print(f"  {method}: {count}")
            
            if "ready_for_analysis" in final_df.columns:
                ready_count = final_df["ready_for_analysis"].sum()
                ready_pct = (ready_count / len(final_df)) * 100
                print(f"\nArticles ready for sentiment analysis: {ready_count} ({ready_pct:.1f}%)")
            
            print("\nColumns added during preprocessing:")
            print("  - cleaned_text: Cleaned version of the article")
            print("  - word_count: Number of words")
            print("  - sentence_count: Number of sentences")
            print("  - avg_word_length: Average word length")
            print("  - financial_term_count: Number of financial terms")
            print("  - money_mentions: Number of money mentions")
            print("  - percentage_mentions: Number of percentage mentions")
            print("  - has_financial_content: True if financial terms found")
            print("  - data_quality_score: Quality score (0-1)")
            print("  - ready_for_analysis: True if suitable for sentiment analysis")
            print("  - processing_date: When this was processed")
            
            if "year" in final_df.columns:
                print("  - year, month, day, day_of_week, hour: Date features")
    else:
        print("\nNo files were processed successfully")

if __name__ == "__main__":
    main()