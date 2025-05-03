# Import required libraries
import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import DebertaTokenizer, DebertaForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import wordnet
from tqdm.notebook import tqdm
import re
import zipfile
import gdown
import datetime
import git
import hashlib
import itertools
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

## Data Pipeline
# Create directory structure
# !mkdir -p data/raw data/processed data/adversarial logs models results

# Function to download and extract datasets
def download_datasets():
    """Download and extract real-world phishing and legitimate email corpora"""
    
    # Download Enron email dataset (legitimate emails)
    print("Downloading Enron dataset...")
    enron_url = f'https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz'
    # !wget -P data/raw {enron_url} -q --show-progress
    # !tar -xzf data/raw/enron_mail_20150507.tar.gz -C data/raw
    
    # Download Nazario phishing corpus
    print("Downloading Nazario phishing corpus...")
    nazario_url = f"https://monkey.org/%7Ejose/phishing/phishing-2024"
    # !wget -P data/raw {nazario_url} -q --show-progress
    # !unzip -q data/raw/phishing-corpora-20221201.zip -d data/raw/phishing
    
    # Download PhishTank dataset (URLs and metadata)
    print("Downloading PhishTank dataset...")
    phishtank_url = f'https://data.phishtank.com/data/online-valid.csv'
    # !wget -P data/raw {phishtank_url} -q --show-progress
    
    # Log dataset metadata
    with open('data/dataset_metadata.json', 'w') as f:
        metadata = {
            'download_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'enron_source': enron_url,
            'nazario_source': nazario_url,
            'phishtank_source': phishtank_url,
            'commit_hash': git.Repo(search_parent_directories=True).head.object.hexsha if os.path.exists('.git') else None
        }
        json.dump(metadata, f, indent=2)
    
    print("All datasets downloaded successfully!")

# Function to preprocess emails
def preprocess_emails():
    """Preprocess raw emails: strip headers, normalize URLs, remove duplicates"""
    
    def extract_email_body(email_text):
        """Extract body from email by removing headers"""
        lines = email_text.split('\n')
        body_start = 0
        
        # Find where headers end and body begins
        for i, line in enumerate(lines):
            if line.strip() == '':
                body_start = i + 1
                break
        
        body = '\n'.join(lines[body_start:])
        return body
    
    def normalize_urls(text):
        """Normalize URLs in text to avoid detection based on specific URLs"""
        # Simple URL normalization - replace actual domains with placeholders
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return re.sub(url_pattern, '[URL]', text)
    
    # Process legitimate emails (Enron)
    print("Processing legitimate emails...")
    legitimate_emails = []
    legitimate_dir = 'data/raw/maildir'
    
    # Sample a subset of Enron folders to keep dataset size manageable
    user_folders = os.listdir(legitimate_dir)
    sampled_users = random.sample(user_folders, min(20, len(user_folders)))
    
    for user in tqdm(sampled_users):
        user_path = os.path.join(legitimate_dir, user)
        if not os.path.isdir(user_path):
            continue
            
        # Walk through user's email folders
        for root, _, files in os.walk(user_path):
            for file in files:
                if len(legitimate_emails) >= 5000:  # Limit the number of legitimate emails
                    break
                    
                try:
                    with open(os.path.join(root, file), 'r', encoding='latin1') as f:
                        email_text = f.read()
                    
                    # Extract body and normalize
                    body = extract_email_body(email_text)
                    body = normalize_urls(body)
                    
                    if len(body.strip()) > 100:  # Filter out very short emails
                        legitimate_emails.append({'text': body, 'label': 0})
                except Exception as e:
                    continue  # Skip problematic files
    
    # Process phishing emails (Nazario)
    print("Processing phishing emails...")
    phishing_emails = []
    phishing_dir = 'data/raw/phishing'
    
    # Walk through phishing corpus folders
    for root, _, files in os.walk(phishing_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.eml'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='latin1') as f:
                        email_text = f.read()
                    
                    # Extract body and normalize
                    body = extract_email_body(email_text)
                    body = normalize_urls(body)
                    
                    if len(body.strip()) > 100:  # Filter out very short emails
                        phishing_emails.append({'text': body, 'label': 1})
                except Exception as e:
                    continue  # Skip problematic files
    
    # Remove duplicates
    print("Removing duplicates...")
    
    def get_email_hash(email):
        """Create a hash of email text to identify duplicates"""
        return hashlib.md5(email['text'].encode()).hexdigest()
    
    unique_emails = {}
    
    for email in legitimate_emails + phishing_emails:
        email_hash = get_email_hash(email)
        if email_hash not in unique_emails:
            unique_emails[email_hash] = email
    
    # Convert to dataframe
    df = pd.DataFrame(list(unique_emails.values()))
    
    # Balance dataset if needed
    min_class_count = min(sum(df['label'] == 0), sum(df['label'] == 1))
    legitimate_df = df[df['label'] == 0].sample(min_class_count, random_state=42)
    phishing_df = df[df['label'] == 1].sample(min_class_count, random_state=42)
    df_balanced = pd.concat([legitimate_df, phishing_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(df_balanced, test_size=0.3, stratify=df_balanced['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # Save processed datasets
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"Dataset processed and split successfully!")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


#  For Kaggle, we can download pre-processed datasets instead of raw ones to save time
def get_processed_datasets(use_cached=True):
    """Get processed datasets, either by loading cached files or processing raw data"""
    if use_cached and os.path.exists('data/processed/train.csv'):
        print("Loading cached datasets...")
        train_df = pd.read_csv('data/processed/train.csv')
        val_df = pd.read_csv('data/processed/val.csv')
        test_df = pd.read_csv('data/processed/test.csv')
    else:
        print("Processing raw datasets...")
        download_datasets()
        train_df, val_df, test_df = preprocess_emails()
    
    return train_df, val_df, test_df

# Let's download a smaller pre-processed dataset for Kaggle
def download_preprocessed_data():
    """Download pre-processed datasets directly for Kaggle"""
    # This would be a link to your publicly accessible pre-processed datasets
    # For the purpose of this notebook, we'll create synthetic data
    
    print("Creating synthetic dataset for demonstration...")
    
    # Download NLTK resources for text generation
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # Create synthetic legitimate emails
    legitimate_templates = [
        "Dear {name}, I hope this email finds you well. Just wanted to follow up on our meeting last week. Let's schedule a call to discuss the project progress. Best regards, {sender}",
        "Hello {name}, Please find attached the quarterly report you requested. Let me know if you need anything else. Thanks, {sender}",
        "Hi team, Reminder about our weekly standup tomorrow at 10AM. Please update your progress on the Jira board. Thanks, {sender}",
        "Good morning {name}, Just checking in regarding the proposal we submitted last month. Do you have any feedback for us? Best, {sender}",
        "Dear colleagues, Please note that the office will be closed next Monday for the holiday. All deadlines remain unchanged. Regards, HR"
    ]
    
    # Create synthetic phishing emails
    phishing_templates = [
        "URGENT: Your account has been compromised. Click here to reset your password immediately: {url}",
        "Dear valued customer, We've noticed suspicious activity on your account. Please verify your identity by clicking this link: {url}",
        "Congratulations! You've won a $1000 Amazon gift card. Claim now at: {url}",
        "Your payment was declined. Update your billing information here: {url} to avoid service interruption.",
        "Security alert: Unusual login detected. If this wasn't you, secure your account immediately: {url}"
    ]
    
    names = ["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "William", "Linda", "David", "Elizabeth"]
    senders = ["Mark", "Sarah", "Alex", "Rachel", "Tom", "Emma", "James", "Sophie", "Daniel", "Olivia"]
    urls = ["http://secure-login.com/verify", "http://account-verification.net/secure", 
            "http://amazon-rewards.com/claim", "http://banking-update.com/renew",
            "http://security-check.org/protect"]
    
    def generate_emails(templates, count, label, is_phishing=False):
        emails = []
        for _ in range(count):
            template = random.choice(templates)
            if is_phishing:
                email = template.format(name=random.choice(names), sender=random.choice(senders), 
                                       url=random.choice(urls))
            else:
                email = template.format(name=random.choice(names), sender=random.choice(senders))
            emails.append({"text": email, "label": label})
        return emails
    
    # Generate balanced dataset
    legitimate_emails = generate_emails(legitimate_templates, 1000, 0)
    phishing_emails = generate_emails(phishing_templates, 1000, 1, is_phishing=True)
    
    all_emails = legitimate_emails + phishing_emails
    random.shuffle(all_emails)
    
    df = pd.DataFrame(all_emails)
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # Save processed datasets
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"Synthetic dataset created and split successfully!")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

# # For Kaggle, use synthetic data for demonstration purposes
# train_df, val_df, test_df = download_preprocessed_data()
