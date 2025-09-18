"""
Tweet Extraction Module

This module handles the extraction of tweet texts from the MFTC dataset using Twitter API v2,
with balanced extraction capabilities to ensure even distribution across moral foundations.
"""

import json
import sys
import time
import requests
from datetime import datetime
import os
from collections import defaultdict, Counter

# Twitter API credentials (should be set in environment or config)
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAEHu4AEAAAAA%2Fkm%2B6TmDldyWwfrG9Xsp5WGp3Po%3DZqdQjus1CY0xe73n0K1Vl5iKrQJKu52k7TPQeWvFKTN9mbsSgL'

def call_twitter_api_v2(tweet_id):
    """
    Use Twitter API v2 to get tweet text.
    
    Args:
        tweet_id: Twitter tweet ID
        
    Returns:
        Tuple of (text, date, success_flag)
    """
    url = f"https://api.twitter.com/2/tweets/{tweet_id}"
    
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}',
        'Content-Type': 'application/json',
    }
    
    params = {
        'tweet.fields': 'created_at,text'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                tweet_data = data['data']
                return tweet_data['text'], tweet_data['created_at'], True
            else:
                print(f"No data returned for tweet {tweet_id}")
                return 'no tweet text available', 'no date available', False
        elif response.status_code == 429:
            print(f"Rate limit exceeded for tweet {tweet_id}")
            return 'rate limit exceeded', 'no date available', False
        elif response.status_code == 404:
            print(f"Tweet {tweet_id} not found (deleted or private)")
            return 'tweet not found', 'no date available', False
        else:
            print(f"Error {response.status_code} for tweet {tweet_id}: {response.text}")
            return 'api error', 'no date available', False
            
    except Exception as e:
        print(f"Exception for tweet {tweet_id}: {e}")
        return 'exception occurred', 'no date available', False

def parse_annotation(annotation_string):
    """Parse annotation string to extract moral foundations."""
    if not isinstance(annotation_string, str):
        return []
    
    foundations = []
    annotation_lower = annotation_string.lower()
    
    # Map common annotation patterns to foundations
    foundation_mapping = {
        'care': 'care',
        'harm': 'care',
        'fairness': 'fairness',
        'cheating': 'fairness',
        'loyalty': 'loyalty',
        'betrayal': 'loyalty',
        'authority': 'authority',
        'subversion': 'authority',
        'sanctity': 'sanctity',
        'degradation': 'sanctity',
        'liberty': 'liberty',
        'oppression': 'liberty'
    }
    
    for key, foundation in foundation_mapping.items():
        if key in annotation_lower:
            foundations.append(foundation)
    
    return foundations

def get_current_distribution():
    """Get current moral foundation distribution from extracted data."""
    file_path = 'data/mftc_extracted_continuous.json'
    
    if not os.path.exists(file_path):
        return Counter()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        foundation_counts = Counter()
        
        for corpus in data:
            tweets = corpus.get('Tweets', [])
            
            for tweet in tweets:
                tweet_text = tweet.get('tweet_text', '')
                
                if tweet_text and tweet_text != 'no tweet text available':
                    # Parse annotations to get moral foundations
                    annotations = tweet.get('annotations', [])
                    tweet_labels = []
                    
                    for annotation in annotations:
                        annotation_text = annotation.get('annotation', '')
                        if annotation_text:
                            foundations = parse_annotation(annotation_text)
                            tweet_labels.extend(foundations)
                    
                    if tweet_labels:
                        # Use the most common label for this tweet
                        most_common_label = max(set(tweet_labels), key=tweet_labels.count)
                        foundation_counts[most_common_label] += 1
        
        return foundation_counts
        
    except Exception as e:
        print(f"Error reading distribution: {e}")
        return Counter()

def get_foundation_priority(current_distribution, tweet_labels):
    """Calculate priority for a tweet based on current distribution."""
    if not tweet_labels:
        return 0
    
    # Use the most common label for this tweet
    most_common_label = max(set(tweet_labels), key=tweet_labels.count)
    
    # Calculate how underrepresented this foundation is
    total_tweets = sum(current_distribution.values())
    if total_tweets == 0:
        return 100  # High priority if no data yet
    
    target_per_foundation = total_tweets / 6  # 6 moral foundations
    current_count = current_distribution.get(most_common_label, 0)
    
    # Higher priority for more underrepresented foundations
    priority = max(0, target_per_foundation - current_count)
    
    return priority

def load_mftc_data():
    """Load the original MFTC dataset."""
    file_path = 'data/mftc.json'
    
    if not os.path.exists(file_path):
        print(f"MFTC data file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading MFTC data: {e}")
        return []

def load_progress():
    """Load extraction progress."""
    file_path = 'data/extraction_progress.json'
    
    if not os.path.exists(file_path):
        return {
            'processed_tweets': [],
            'current_corpus': 0,
            'current_tweet': 0,
            'successful_extractions': 0
        }
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading progress: {e}")
        return {
            'processed_tweets': [],
            'current_corpus': 0,
            'current_tweet': 0,
            'successful_extractions': 0
        }

def save_progress(progress):
    """Save extraction progress."""
    file_path = 'data/extraction_progress.json'
    
    try:
        with open(file_path, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")

def load_extracted_data():
    """Load existing extracted data."""
    file_path = 'data/mftc_extracted_continuous.json'
    
    if not os.path.exists(file_path):
        # Initialize with empty structure
        return [
            {"Corpus": "ALM", "Tweets": []},
            {"Corpus": "Baltimore", "Tweets": []},
            {"Corpus": "BLM", "Tweets": []},
            {"Corpus": "Davidson", "Tweets": []},
            {"Corpus": "Election", "Tweets": []},
            {"Corpus": "MeToo", "Tweets": []},
            {"Corpus": "Sandy", "Tweets": []}
        ]
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading extracted data: {e}")
        return [
            {"Corpus": "ALM", "Tweets": []},
            {"Corpus": "Baltimore", "Tweets": []},
            {"Corpus": "BLM", "Tweets": []},
            {"Corpus": "Davidson", "Tweets": []},
            {"Corpus": "Election", "Tweets": []},
            {"Corpus": "MeToo", "Tweets": []},
            {"Corpus": "Sandy", "Tweets": []}
        ]

def save_extracted_data(extracted_data):
    """Save extracted data."""
    file_path = 'data/mftc_extracted_continuous.json'
    
    try:
        with open(file_path, 'w') as f:
            json.dump(extracted_data, f, indent=2)
    except Exception as e:
        print(f"Error saving extracted data: {e}")

def balanced_extraction():
    """
    Continuously extract tweets, making up to 15 requests every 15 minutes,
    saving progress incrementally and prioritizing underrepresented moral foundations.
    """
    print("Starting balanced tweet extraction...")
    print("This will run indefinitely, making up to 15 requests every 15 minutes")
    print("Press Ctrl+C to stop")
    
    mftc_data = load_mftc_data()
    progress = load_progress()
    extracted_data = load_extracted_data()
    
    processed_tweets = set(progress['processed_tweets'])
    
    # Count existing foundations in extracted data and add to processed tweets
    print("Analyzing existing data...")
    current_distribution = get_current_distribution()
    
    print(f"Found {len(processed_tweets)} already processed tweets in existing data")
    
    print("Current foundation distribution:")
    total_tweets = sum(current_distribution.values())
    for foundation, count in current_distribution.items():
        percentage = (count / total_tweets * 100) if total_tweets > 0 else 0
        print(f"  {foundation}: {count} ({percentage:.1f}%)")
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{current_time}] Starting balanced extraction cycle...")
            
            # Collect tweets with their priorities
            tweet_candidates = []
            
            corpus_idx = progress['current_corpus']
            tweet_idx = progress['current_tweet']
            
            # Scan through tweets to find candidates
            while corpus_idx < len(mftc_data):
                corpus = mftc_data[corpus_idx]
                
                # Skip Davidson corpus
                if corpus["Corpus"] == "Davidson":
                    corpus_idx += 1
                    tweet_idx = 0
                    continue
                
                if tweet_idx < len(corpus["Tweets"]):
                    tweet = corpus["Tweets"][tweet_idx]
                    tweet_id = tweet["tweet_id"]
                    
                    if tweet_id not in processed_tweets:
                        # Calculate priority for this tweet
                        annotations = tweet.get('annotations', [])
                        tweet_labels = []
                        for annotation in annotations:
                            annotation_text = annotation.get('annotation', '')
                            if annotation_text:
                                foundations = parse_annotation(annotation_text)
                                tweet_labels.extend(foundations)
                        
                        priority = get_foundation_priority(current_distribution, tweet_labels)
                        
                        tweet_candidates.append({
                            'corpus_idx': corpus_idx,
                            'tweet_idx': tweet_idx,
                            'tweet_id': tweet_id,
                            'annotations': tweet["annotations"],
                            'corpus_name': corpus["Corpus"],
                            'priority': priority,
                            'labels': tweet_labels # Store labels for updating counts
                        })
                    
                    tweet_idx += 1
                else:
                    corpus_idx += 1
                    tweet_idx = 0
            
            if not tweet_candidates:
                print("All tweets have been processed!")
                break
            
            # Sort by priority (highest first)
            tweet_candidates.sort(key=lambda x: x['priority'], reverse=True)
            
            # Process up to 15 tweets per cycle
            tweets_this_cycle = 0
            max_tweets_per_cycle = 15
            
            for candidate in tweet_candidates:
                if tweets_this_cycle >= max_tweets_per_cycle:
                    break
                
                # Double-check: Skip if already processed (safety check)
                if candidate['tweet_id'] in processed_tweets:
                    print(f"Skipping already processed tweet: {candidate['tweet_id']}")
                    tweets_this_cycle += 1
                    continue
                
                # Process the tweet
                print(f"Processing tweet {candidate['tweet_id']} from {candidate['corpus_name']} (priority: {candidate['priority']:.1f})")
                
                text, date, success = call_twitter_api_v2(candidate['tweet_id'])
                
                # Add to extracted data
                extracted_tweet = {
                    "tweet_id": candidate['tweet_id'],
                    "tweet_text": text,
                    "date": date,
                    "annotations": candidate['annotations']
                }
                
                # Find the corresponding corpus in extracted data
                for extracted_corpus in extracted_data:
                    if extracted_corpus["Corpus"] == candidate['corpus_name']:
                        extracted_corpus["Tweets"].append(extracted_tweet)
                        break
                
                # Update progress
                processed_tweets.add(candidate['tweet_id'])
                progress['processed_tweets'] = list(processed_tweets)
                progress['current_corpus'] = candidate['corpus_idx']
                progress['current_tweet'] = candidate['tweet_idx'] + 1
                
                if success:
                    progress['successful_extractions'] += 1
                    print(f"SUCCESS: Extracted text: {text[:100]}...")
                    
                    # Update foundation counts
                    if candidate['labels']:
                        most_common_label = max(set(candidate['labels']), key=candidate['labels'].count)
                        current_distribution[most_common_label] += 1
                else:
                    print(f"FAILED: Could not extract text")
                
                tweets_this_cycle += 1
                
                time.sleep(2) # Small delay between requests
            
            save_progress(progress)
            save_extracted_data(extracted_data)
            
            print(f"Cycle complete: {tweets_this_cycle} tweets processed this cycle")
            print(f"Total progress: {len(processed_tweets)} tweets processed, {progress['successful_extractions']} successful")
            print("Current foundation distribution:")
            total_tweets = sum(current_distribution.values())
            for foundation, count in current_distribution.items():
                percentage = (count / total_tweets * 100) if total_tweets > 0 else 0
                print(f"  {foundation}: {count} ({percentage:.1f}%)")
            print(f"Waiting 15 minutes until next cycle...")
            
            # Wait 15 minutes until next cycle
            for i in range(900):
                time.sleep(1)
                if i % 60 == 0 and i > 0:  # Print every minute
                    remaining = 900 - i
                    print(f"Waiting... {remaining} seconds remaining")
    
    except KeyboardInterrupt:
        print("\nStopping extraction...")
        print(f"Final progress: {len(processed_tweets)} tweets processed, {progress['successful_extractions']} successful")
        print("Final foundation distribution:")
        total_tweets = sum(current_distribution.values())
        for foundation, count in current_distribution.items():
            percentage = (count / total_tweets * 100) if total_tweets > 0 else 0
            print(f"  {foundation}: {count} ({percentage:.1f}%)")
        print("Data saved to data/mftc_extracted_continuous.json")
        print("Progress saved to data/extraction_progress.json")

def main():
    """Main function."""
    balanced_extraction()

if __name__ == "__main__":
    main()