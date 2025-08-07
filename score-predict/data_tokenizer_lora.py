import json
import os
import glob
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import numpy as np

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

def create_lora_model(model_name='roberta-base', lora_r=8, lora_alpha=32, lora_dropout=0.1):
    """Create a LoRA-enabled model for fine-tuning."""
    
    # Load the base model
    base_model = RobertaModel.from_pretrained(model_name)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # or TaskType.SEQ_CLS for classification
        inference_mode=False,
        r=lora_r,  # rank
        lora_alpha=lora_alpha,  # scaling parameter
        lora_dropout=lora_dropout,
        target_modules=["query", "value", "key", "dense"],  # RoBERTa attention modules
        bias="none",  # or "all" if you want to adapt bias terms too
    )
    
    # Create LoRA model
    lora_model = get_peft_model(base_model, peft_config)
    
    print(f"LoRA model created with {lora_model.num_parameters()} total parameters")
    print(f"Trainable parameters: {lora_model.get_nb_trainable_parameters()}")
    
    return lora_model, peft_config

class RedditDataset(torch.utils.data.Dataset):
    """Custom Dataset class for Reddit data compatible with LoRA training."""
    
    def __init__(self, tokens, upvote_ratios, over_18_flags, subreddits, target_scores):
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.upvote_ratios = upvote_ratios
        self.over_18_flags = over_18_flags
        self.subreddits = subreddits
        self.target_scores = target_scores
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'upvote_ratio': self.upvote_ratios[idx],
            'over_18': self.over_18_flags[idx],
            'subreddit': self.subreddits[idx],
            'labels': self.target_scores[idx]  # Using 'labels' as standard for HuggingFace
        }

def prepare_data_for_lora(max_length=512, test_split=0.2, random_seed=42):
    """Prepare and return all processed data needed for LoRA training."""
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
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
        # Extract subreddit from permalink
        permalink = submission.get('permalink', '')
        try:
            subreddit = permalink.split('/')[2]
        except IndexError:
            subreddit = 'unknown'

        # Get top comments for this submission (optional enhancement)
        submission_comments = comments_by_submission.get(submission.get('id', ''), [])
        top_comment_text = ""
        if submission_comments:
            # Get the top comment by score
            top_comment = max(submission_comments, key=lambda x: x.get('score', 0))
            top_comment_text = f" [TOP_COMMENT] {top_comment.get('body', '')}"

        # Combine all features into a single dictionary
        combined_features = {
            'submission_id': submission.get('id', ''),
            'title': submission.get('title', ''),
            'selftext': submission.get('selftext', ''),
            'top_comment': top_comment_text,
            'subreddit': subreddit,
            'upvote_ratio': submission.get('upvote_ratio', 0.0),
            'over_18': 1 if submission.get('over_18', False) else 0,
            'target_score': submission.get('score', 0)
        }
        processed_data.append(combined_features)

    print("Preprocessing complete.")

    # Tokenization with LoRA-optimized settings
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Add special tokens if needed
    special_tokens = ['[TOP_COMMENT]']
    tokenizer.add_tokens(special_tokens)

    # Prepare text input - combining title, selftext, and top comment
    text_input = []
    for item in processed_data:
        combined_text = f"{item['title']} {item['selftext']} {item['top_comment']}"
        text_input.append(combined_text.strip())

    # Tokenize with appropriate max_length for LoRA efficiency
    tokens = tokenizer(
        text_input, 
        padding=True, 
        truncation=True, 
        max_length=max_length,
        return_tensors="pt"
    )

    # Create subreddit encoder
    subreddit_encoder = LabelEncoder()
    subreddit_names = [item['subreddit'] for item in processed_data]
    encoded_subreddits = subreddit_encoder.fit_transform(subreddit_names)
    
    print(f"Found {len(subreddit_encoder.classes_)} unique subreddits")

    # Prepare numerical and categorical inputs
    upvote_ratios = torch.tensor([item['upvote_ratio'] for item in processed_data], dtype=torch.float).unsqueeze(1)
    over_18_flags = torch.tensor([item['over_18'] for item in processed_data], dtype=torch.float).unsqueeze(1)
    subreddits = torch.tensor(encoded_subreddits, dtype=torch.long)

    # Prepare and normalize target scores
    target_scores = [item['target_score'] for item in processed_data]
    score_scaler = StandardScaler()
    normalized_scores = score_scaler.fit_transform([[score] for score in target_scores])
    normalized_scores = torch.tensor(normalized_scores, dtype=torch.float).squeeze()

    # Create train/test split
    n_samples = len(processed_data)
    indices = torch.randperm(n_samples)
    split_idx = int(n_samples * (1 - test_split))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Split all data
    def split_data(data, train_idx, test_idx):
        if isinstance(data, torch.Tensor):
            return data[train_idx], data[test_idx]
        elif isinstance(data, dict):
            train_data = {k: v[train_idx] for k, v in data.items()}
            test_data = {k: v[test_idx] for k, v in data.items()}
            return train_data, test_data
        else:
            return data  # For scalers and encoders
    
    train_tokens, test_tokens = split_data(tokens, train_indices, test_indices)
    train_upvote_ratios, test_upvote_ratios = split_data(upvote_ratios, train_indices, test_indices)
    train_over_18_flags, test_over_18_flags = split_data(over_18_flags, train_indices, test_indices)
    train_subreddits, test_subreddits = split_data(subreddits, train_indices, test_indices)
    train_scores, test_scores = split_data(normalized_scores, train_indices, test_indices)
    
    # Create datasets
    train_dataset = RedditDataset(train_tokens, train_upvote_ratios, train_over_18_flags, 
                                train_subreddits, train_scores)
    test_dataset = RedditDataset(test_tokens, test_upvote_ratios, test_over_18_flags, 
                               test_subreddits, test_scores)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return (tokenizer, train_dataset, test_dataset, score_scaler, subreddit_encoder,
            len(subreddit_encoder.classes_))

def prepare_lora_training_setup(model_name='roberta-base', lora_r=8, lora_alpha=32, 
                              lora_dropout=0.1, max_length=512):
    """Complete setup for LoRA training including model and data preparation."""
    
    # Prepare data
    (tokenizer, train_dataset, test_dataset, score_scaler, 
     subreddit_encoder, num_subreddits) = prepare_data_for_lora(max_length=max_length)
    
    # Create LoRA model
    lora_model, peft_config = create_lora_model(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # Resize token embeddings if we added special tokens
    lora_model.resize_token_embeddings(len(tokenizer))
    
    return {
        'model': lora_model,
        'peft_config': peft_config,
        'tokenizer': tokenizer,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'score_scaler': score_scaler,
        'subreddit_encoder': subreddit_encoder,
        'num_subreddits': num_subreddits
    }

# For backward compatibility and testing
if __name__ == "__main__":
    print("Setting up LoRA training environment...")
    
    try:
        setup = prepare_lora_training_setup(
            model_name='roberta-base',
            lora_r=16,  # Higher rank for better performance
            lora_alpha=32,
            lora_dropout=0.1,
            max_length=256  # Shorter for faster training
        )
        
        print("✅ LoRA setup complete!")
        print(f"Model type: {type(setup['model'])}")
        print(f"Training samples: {len(setup['train_dataset'])}")
        print(f"Test samples: {len(setup['test_dataset'])}")
        print(f"Vocabulary size: {len(setup['tokenizer'])}")
        print(f"Number of subreddits: {setup['num_subreddits']}")
        
        # Test a single batch
        sample_batch = setup['train_dataset'][0]
        print(f"\nSample batch keys: {sample_batch.keys()}")
        print(f"Input shape: {sample_batch['input_ids'].shape}")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install peft transformers torch scikit-learn")
    except Exception as e:
        print(f"❌ Error during setup: {e}")