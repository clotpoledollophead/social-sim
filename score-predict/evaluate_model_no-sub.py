import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaModel, RobertaTokenizer
import joblib
import json
import glob
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from datetime import datetime

class RobertaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.numerical_proj = nn.Linear(2, 64)
        self.regressor = nn.Sequential(
            nn.Linear(768 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = roberta_output.last_hidden_state[:, 0, :]
        num_proj = self.numerical_proj(numerical_features)
        combined = torch.cat((cls_embedding, num_proj), dim=1)
        return self.regressor(combined)

def load_json_lines(file_path):
    """Load JSON lines file (same as your original function)."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
    return data

def load_model_and_scaler(model_path="trained_models/roberta_score_regressor_withoutsub-3epochs.pt", scaler_path="trained_models/score_scaler_withoutsub-3epochs.save"):
    """Load the trained model and scaler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = RobertaRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    return model, scaler, tokenizer, device

def evaluate_from_json_files(scrapes_dir="../scraping/scrapes/", batch_size=32, save_results=True):
    """
    Evaluate posts from your original JSON files format.
    
    Args:
        scrapes_dir (str): Directory containing your *_submissions.json files
        batch_size (int): Batch size for processing
        save_results (bool): Whether to save detailed results
    
    Returns:
        pd.DataFrame: Results with predictions and errors
    """
    
    print(f"🔍 Looking for JSON files in: {scrapes_dir}")
    
    # Load model and scaler
    print("Loading trained model...")
    model, scaler, tokenizer, device = load_model_and_scaler()
    
    # Find and load all submission files
    all_submissions_data = []
    all_comments_data = []
    
    submission_files = glob.glob(os.path.join(scrapes_dir, "*_submissions.json"))
    comment_files = glob.glob(os.path.join(scrapes_dir, "*_comments.json"))
    
    print(f"📁 Found {len(submission_files)} submission files")
    print(f"📁 Found {len(comment_files)} comment files")
    
    # Load submissions
    for file in submission_files:
        print(f"Loading submissions from: {os.path.basename(file)}")
        data = load_json_lines(file)
        all_submissions_data.extend(data)
        print(f"  → Loaded {len(data)} submissions")
    
    # Load comments (if you want to use them later)
    for file in comment_files:
        print(f"Loading comments from: {os.path.basename(file)}")
        data = load_json_lines(file)
        all_comments_data.extend(data)
        print(f"  → Loaded {len(data)} comments")
    
    print(f"\n📊 Total loaded: {len(all_submissions_data)} submissions, {len(all_comments_data)} comments")
    
    if len(all_submissions_data) == 0:
        print("❌ No submission data found! Check your file paths.")
        return None
    
    # Process submissions (same logic as your original data_tokenizer.py)
    print("🔄 Processing submissions for evaluation...")
    
    processed_posts = []
    for submission in all_submissions_data:
        # Extract subreddit from permalink
        permalink = submission.get('permalink', '')
        try:
            subreddit = permalink.split('/')[2]
        except IndexError:
            subreddit = 'unknown'
        
        post = {
            'submission_id': submission.get('id', ''),
            'title': submission.get('title', ''),
            'selftext': submission.get('selftext', ''),
            'subreddit': subreddit,
            'upvote_ratio': submission.get('upvote_ratio', 0.5),
            'over_18': submission.get('over_18', False),
            'actual_score': submission.get('score', 0)  # This is what we'll compare against
        }
        processed_posts.append(post)
    
    print(f"✅ Processed {len(processed_posts)} posts ready for evaluation")
    
    # Now run predictions in batches
    print("🚀 Running predictions...")
    
    all_results = []
    
    # Process in batches
    for i in tqdm(range(0, len(processed_posts), batch_size), desc="Predicting scores"):
        batch_posts = processed_posts[i:i + batch_size]
        
        # Prepare batch data
        texts = []
        upvote_ratios = []
        over_18_flags = []
        
        for post in batch_posts:
            text = post['title'] + " " + post['selftext']
            texts.append(text)
            upvote_ratios.append(post['upvote_ratio'])
            over_18_flags.append(1.0 if post['over_18'] else 0.0)
        
        # Tokenize
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Prepare numerical features
        upvote_tensor = torch.tensor(upvote_ratios, dtype=torch.float).unsqueeze(1)
        over_18_tensor = torch.tensor(over_18_flags, dtype=torch.float).unsqueeze(1)
        numerical_features = torch.cat([upvote_tensor, over_18_tensor], dim=1)
        
        # Move to device
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        numerical_features = numerical_features.to(device)
        
        # Predict
        with torch.no_grad():
            predictions = model(input_ids, attention_mask, numerical_features)
            predictions = predictions.squeeze().cpu().numpy()
            
            # Handle single prediction case
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            
            # Denormalize predictions
            denorm_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Store results
        for j, (post, pred) in enumerate(zip(batch_posts, denorm_predictions)):
            result = {
                'submission_id': post['submission_id'],
                'title': post['title'],
                'selftext_preview': post['selftext'][:100] + ('...' if len(post['selftext']) > 100 else ''),
                'subreddit': post['subreddit'],
                'upvote_ratio': post['upvote_ratio'],
                'over_18': post['over_18'],
                'actual_score': post['actual_score'],
                'predicted_score': round(float(pred), 1),
                'error': abs(post['actual_score'] - pred),
                'relative_error': abs(post['actual_score'] - pred) / max(abs(post['actual_score']), 1) * 100
            }
            all_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate overall metrics
    actual_scores = results_df['actual_score'].values
    predicted_scores = results_df['predicted_score'].values
    
    mse = mean_squared_error(actual_scores, predicted_scores)
    mae = mean_absolute_error(actual_scores, predicted_scores)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_scores, predicted_scores)
    
    print(f"\n📈 EVALUATION RESULTS:")
    print(f"=" * 40)
    print(f"Total posts evaluated: {len(results_df):,}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean actual score: {np.mean(actual_scores):.2f}")
    print(f"Mean predicted score: {np.mean(predicted_scores):.2f}")
    
    # # Show some examples
    # print(f"\n🎯 SAMPLE PREDICTIONS:")
    # print(f"=" * 80)
    # sample_results = results_df.sample(min(10, len(results_df)))
    # for _, row in sample_results.iterrows():
    #     print(f"'{row['title'][:50]}...'")
    #     print(f"  Actual: {row['actual_score']:.0f} | Predicted: {row['predicted_score']:.1f} | Error: {row['error']:.1f}")
    #     print()
    
    # # Show best and worst predictions
    # print(f"🏆 BEST PREDICTIONS (lowest error):")
    # best_predictions = results_df.nsmallest(5, 'error')
    # for _, row in best_predictions.iterrows():
    #     print(f"  Error {row['error']:.1f}: '{row['title'][:60]}...' (actual: {row['actual_score']}, pred: {row['predicted_score']:.1f})")
    
    # print(f"\n💥 WORST PREDICTIONS (highest error):")
    # worst_predictions = results_df.nlargest(5, 'error')
    # for _, row in worst_predictions.iterrows():
    #     print(f"  Error {row['error']:.1f}: '{row['title'][:60]}...' (actual: {row['actual_score']}, pred: {row['predicted_score']:.1f})")
    
    # Subreddit analysis
    print(f"\n📱 TOP SUBREDDITS BY PREDICTION ERROR:")
    subreddit_stats = results_df.groupby('subreddit').agg({
        'error': ['mean', 'count'],
        'actual_score': 'mean',
        'predicted_score': 'mean'
    }).round(2)
    
    # Flatten column names
    subreddit_stats.columns = ['avg_error', 'post_count', 'avg_actual', 'avg_predicted']
    top_subreddits = subreddit_stats[subreddit_stats['post_count'] >= 10].sort_values('avg_error', ascending=False).head(10)
    
    for subreddit, stats in top_subreddits.iterrows():
        print(f"  r/{subreddit}: {stats['post_count']:.0f} posts, avg error: {stats['avg_error']:.1f}")
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'total_posts': len(results_df),
            'metrics': {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            },
            'files_processed': {
                'submission_files': [os.path.basename(f) for f in submission_files],
                'comment_files': [os.path.basename(f) for f in comment_files]
            }
        }
        
        summary_file = f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Summary saved to: {summary_file}")
    
    return results_df

if __name__ == "__main__":
    print("🚀 Reddit Score Evaluator - JSON Files")
    print("=" * 50)
    
    # Get directory path
    scrapes_dir = input("Enter path to scrapes directory (default: ../scraping/scrapes/): ").strip()
    if not scrapes_dir:
        scrapes_dir = "../scraping/scrapes/"
    
    # Get batch size
    try:
        batch_size = int(input("Enter batch size (default: 32): ").strip() or "32")
    except ValueError:
        batch_size = 32
    
    print(f"\nStarting evaluation...")
    print(f"Directory: {scrapes_dir}")
    print(f"Batch size: {batch_size}")
    
    # Run evaluation
    results = evaluate_from_json_files(
        scrapes_dir=scrapes_dir,
        batch_size=batch_size,
        save_results=True
    )
    
    if results is not None:
        print(f"\n✅ Evaluation complete! Check the saved files for detailed results.")
    else:
        print(f"\n❌ Evaluation failed. Please check your file paths and try again.")