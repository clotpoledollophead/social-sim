import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import RobertaModel
from tqdm import tqdm
from data_tokenizer import prepare_data
import joblib
import os

print("Preparing data... \n ૮₍ ´ ꒳ `₎ა \n")

# Enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data (now includes subreddit encoder)
tokenizer, tokenized, upvote_ratios, over_18_flags, subreddits, normalized_scores, score_scaler, subreddit_encoder = prepare_data()

print(f"Number of unique subreddits: {len(subreddit_encoder.classes_)}")

# Combine features
input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']
numerical_features = torch.cat([upvote_ratios, over_18_flags], dim=1)

# Dataset & Split (now includes subreddit data)
dataset = TensorDataset(input_ids, attention_mask, numerical_features, subreddits, normalized_scores)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")

# Updated Model with Subreddit Embedding
class RobertaRegressor(nn.Module):
    def __init__(self, num_subreddits, embedding_dim=32):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta.train()  # Unfreeze
        
        # Subreddit embedding layer
        self.subreddit_embedding = nn.Embedding(num_subreddits, embedding_dim)
        
        # Numerical features projection (upvote_ratio + over_18)
        self.numerical_proj = nn.Linear(2, 64)
        
        # Updated regressor to handle RoBERTa + numerical + subreddit features
        self.regressor = nn.Sequential(
            nn.Linear(768 + 64 + embedding_dim, 128),  # RoBERTa + numerical + subreddit
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, numerical_features, subreddit_ids):
        # RoBERTa output
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = roberta_output.last_hidden_state[:, 0, :]
        
        # Numerical features projection
        num_proj = self.numerical_proj(numerical_features)
        
        # Subreddit embedding
        subreddit_emb = self.subreddit_embedding(subreddit_ids)
        
        # Combine all features
        combined = torch.cat((cls_embedding, num_proj, subreddit_emb), dim=1)
        return self.regressor(combined)

# Initialize model with number of subreddits
num_subreddits = len(subreddit_encoder.classes_)
model = RobertaRegressor(num_subreddits=num_subreddits).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

print(f"Model initialized with {num_subreddits} subreddits")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Validation function
def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, num_feats, subreddit_ids, targets in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            num_feats = num_feats.to(device)
            subreddit_ids = subreddit_ids.to(device)
            targets = targets.to(device)
            
            preds = model(input_ids, attention_mask, num_feats, subreddit_ids)
            loss = loss_fn(preds.squeeze(), targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# Training loop
EPOCHS = 3
best_val_loss = float('inf')

print(f"\nStarting training for {EPOCHS} epochs...")
print("=" * 50)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    
    for input_ids, attention_mask, num_feats, subreddit_ids, targets in loop:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        num_feats = num_feats.to(device)
        subreddit_ids = subreddit_ids.to(device)
        targets = targets.to(device)

        preds = model(input_ids, attention_mask, num_feats, subreddit_ids)
        loss = loss_fn(preds.squeeze(), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # Validation
    val_loss = validate(model, val_loader, loss_fn, device)
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Training Loss: {avg_train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"  ✅ New best validation loss! Saving model...")
        
        # Create directory if it doesn't exist
        os.makedirs("trained_models", exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), "trained_models/roberta_score_regressor_withsub.pt")
        
        # Save scaler
        joblib.dump(score_scaler, "trained_models/score_scaler_withsub.save")
        
        # Save subreddit encoder
        joblib.dump(subreddit_encoder, "trained_models/subreddit_encoder_withsub.save")
    
    print("-" * 30)

print(f"\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Files saved:")
print(f"  - trained_models/roberta_score_regressor_withsub.pt")
print(f"  - trained_models/score_scaler_withsub.save")
print(f"  - trained_models/subreddit_encoder_withsub.save")
print("\n ૮₍ ´ ꒳ `₎ა Training complete!")

# Optional: Test set evaluation
print("\nEvaluating on test set...")
test_loader = DataLoader(test_set, batch_size=8)
test_loss = validate(model, test_loader, loss_fn, device)
print(f"Test Loss: {test_loss:.4f}")

# Print some subreddit information
print(f"\nSubreddit information:")
print(f"Total unique subreddits: {len(subreddit_encoder.classes_)}")
print(f"First 10 subreddits: {list(subreddit_encoder.classes_[:10])}")
if len(subreddit_encoder.classes_) > 10:
    print(f"... and {len(subreddit_encoder.classes_) - 10} more")