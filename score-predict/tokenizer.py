import json
import os
import glob
import torch
from transformers import RobertaTokenizer

def load_json_lines(file_path):
    """Loads a JSON file where each line is a separate JSON object."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
    return data

# Load data from files

scrapes_dir = "../scraping/scrapes/"

all_submissions_data = []
all_comments_data = []

submission_files = glob.glob(os.path.join(scrapes_dir, "*_submissions.json"))
for file in submission_files:
    print(f"Loading submissions from: {file}")
    all_submissions_data.extend(load_json_lines(file))

comment_files = glob.glob(os.path.join(scrapes_dir, "*_comments.json"))
for file in comment_files:
    print(f"Loading comments from: {file}")
    all_comments_data.extend(load_json_lines(file))

print(f"Loaded {len(all_submissions_data)} submissions and {len(all_comments_data)} comments.")

# Create a dictionary for quick lookup of comments by submission_id
comments_by_submission = {}
for comment in all_comments_data:
    submission_id = comment.get('submission_id')
    if submission_id:
        if submission_id not in comments_by_submission:
            comments_by_submission[submission_id] = []
        comments_by_submission[submission_id].append(comment)

# Process submissions and combine with comments
processed_data = []
for submission in all_submissions_data:
    submission_id = submission['id']
    submission_comments = comments_by_submission.get(submission_id, [])

    # Combine all features into a single dictionary
    combined_features = {
        'submission_id': submission_id,
        'title': submission.get('title', ''),
        'selftext': submission.get('selftext', ''),
        'subreddit': 'r/psychology',  # Assumed from file name, but could be extracted if varied
        'upvote_ratio': submission.get('upvote_ratio', 0.0),
        'over_18': 1 if submission.get('over_18', False) else 0,
        'target_score': submission.get('score', 0)
    }
    processed_data.append(combined_features)

print("Preprocessing complete. Processed data:")
print(processed_data)

# Tokenization
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

text_input = [item['title'] + " " + item['selftext'] for item in processed_data]
tokens = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")

# Prepare numerical and categorical inputs
upvote_ratios = torch.tensor([item['upvote_ratio'] for item in processed_data], dtype=torch.float)
over_18_flags = torch.tensor([item['over_18'] for item in processed_data], dtype=torch.float)

# Prepare target scores
target_scores = torch.tensor([item['target_score'] for item in processed_data], dtype=torch.float)

print("\nTokenized text input:")
print(tokens)
print("\nOther features:")
print("Upvote Ratios:", upvote_ratios)
print("Over 18 Flags:", over_18_flags)
print("\nTarget Scores:", target_scores)
