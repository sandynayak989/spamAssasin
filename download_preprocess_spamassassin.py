import os
import tarfile
import pandas as pd
from pathlib import Path
import re
import shutil

# Define directories and files
SPAM_PATH = os.path.join("datasets", "spam")
FILES = {
    "20030228_easy_ham.tar.bz2": {"label": "ham", "folder": "easy_ham"},
    "20030228_spam.tar.bz2": {"label": "spam", "folder": "spam"}
}

# Function to extract tarballs
def extract_spam_data(spam_path=SPAM_PATH):
    for filename in FILES.keys():
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            print(f"File {path} not found. Please ensure tarball is in {spam_path}")
            continue
        print(f"Extracting {filename}...")
        try:
            with tarfile.open(path, mode='r:bz2') as tar:
                tar.extractall(path=spam_path, filter='data')
            print(f"Extracted to {spam_path}")
        except Exception as e:
            print(f"Error extracting {filename}: {e}")

# Function to clean email text
def clean_email_text(text):
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to process emails and create DataFrame
def create_email_dataframe(spam_path=SPAM_PATH):
    emails = []
    labels = []
    
    for filename, info in FILES.items():
        folder_name = info["folder"]
        folder_path = os.path.join(spam_path, folder_name)
        print(f"Checking folder: {folder_path}, exists: {os.path.exists(folder_path)}")
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} not found. Ensure extraction was successful.")
            continue
        print(f"Processing folder: {folder_path}")
        for email_file in Path(folder_path).glob("*"):
            if email_file.is_file():
                try:
                    with open(email_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        cleaned_content = clean_email_text(content)
                        emails.append(cleaned_content)
                        labels.append(info["label"])
                except Exception as e:
                    print(f"Error reading {email_file}: {e}")
    
    if not emails:
        print("No emails processed. Check folder structure and file integrity.")
        return pd.DataFrame({'text': [], 'label': []})
    
    return pd.DataFrame({'text': emails, 'label': labels})

# Main execution
if __name__ == "__main__":
    # Clean up existing folders to avoid conflicts
    for filename, info in FILES.items():
        folder_path = os.path.join(SPAM_PATH, info["folder"])
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    
    # Step 1: Extract dataset
    extract_spam_data()
    
    # Step 2: Process emails and create DataFrame
    df = create_email_dataframe()
    
    # Step 3: Save to CSV
    output_csv = "enron_spam_data.csv"
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")
    print(f"Total emails: {len(df)}")
    print(f"Spam emails: {sum(df['label'] == 'spam')}")
    print(f"Ham emails: {sum(df['label'] == 'ham')}")