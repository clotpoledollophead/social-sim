import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
from tqdm import tqdm
from data_tokenizer_lora import prepare_lora_training_setup, create_lora_model
import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

print("Preparing LoRA training environment... \n ૮₍ ´ ꒳ `₎ა \n")

# Enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# LoRA configuration
LORA_CONFIG = {
    'lora_r': 16,           # Rank - higher = more capacity but slower
    'lora_alpha': 32,       # Scaling factor
    'lora_dropout': 0.1,    # Regularization
    'max_length': 256       # Token limit for efficiency
}

# Load data with LoRA setup
print("Loading and preparing data...")
setup = prepare_lora_training_setup(
    model_name='roberta-base',
    **LORA_CONFIG
)

tokenizer = setup['tokenizer']
train_dataset = setup['train_dataset']
test_dataset = setup['test_dataset']
score_scaler = setup['score_scaler']
subreddit_encoder = setup['subreddit_encoder']
num_subreddits = setup['num_subreddits']

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of unique subreddits: {num_subreddits}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Enhanced LoRA Model with Multi-Modal Features
class LoRAMultiModalRegressor(nn.Module):
    def __init__(self, lora_model, num_subreddits, embedding_dim=32):
        super().__init__()
        self.lora_roberta = lora_model
        
        # Freeze base model parameters - only LoRA adapters will be trained
        for param in self.lora_roberta.base_model.parameters():
            param.requires_grad = False
            
        # Enable LoRA adapter parameters
        for name, param in self.lora_roberta.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
        # Additional feature processing layers
        self.subreddit_embedding = nn.Embedding(num_subreddits, embedding_dim)
        self.numerical_proj = nn.Linear(2, 64)  # upvote_ratio + over_18
        
        # Multi-modal fusion and regression head
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 + 64 + embedding_dim, 256),  # RoBERTa + numerical + subreddit
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, upvote_ratio, over_18, subreddit):
        # Get LoRA-enhanced RoBERTa embeddings
        roberta_output = self.lora_roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = roberta_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Process numerical features
        numerical_features = torch.cat([upvote_ratio, over_18], dim=1)
        num_proj = self.numerical_proj(numerical_features)
        
        # Process subreddit embedding
        subreddit_emb = self.subreddit_embedding(subreddit)
        
        # Combine all modalities
        combined_features = torch.cat((cls_embedding, num_proj, subreddit_emb), dim=1)
        fused = self.fusion_layer(combined_features)
        
        # Final prediction
        return self.regressor(fused)
    
    def get_trainable_parameters(self):
        """Return count of trainable parameters"""
        lora_params = sum(p.numel() for p in self.lora_roberta.parameters() if p.requires_grad)
        additional_params = sum(p.numel() for n, p in self.named_parameters() 
                             if 'lora_roberta' not in n and p.requires_grad)
        return lora_params + additional_params

# Initialize LoRA model
print("Initializing LoRA model...")
base_lora_model, peft_config = create_lora_model(
    model_name='roberta-base',
    **{k: v for k, v in LORA_CONFIG.items() if k != 'max_length'}
)

# Resize embeddings for any special tokens
base_lora_model.resize_token_embeddings(len(tokenizer))

# Create complete model
model = LoRAMultiModalRegressor(
    lora_model=base_lora_model,
    num_subreddits=num_subreddits,
    embedding_dim=32
).to(device)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = model.get_trainable_parameters()
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

# Optimizer and loss
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], 
    lr=5e-4,  # Higher LR for LoRA
    weight_decay=0.01
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

loss_fn = nn.MSELoss()

# Validation function with metrics
def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            upvote_ratio = batch['upvote_ratio'].to(device)
            over_18 = batch['over_18'].to(device)
            subreddit = batch['subreddit'].to(device)
            labels = batch['labels'].to(device)
            
            preds = model(input_ids, attention_mask, upvote_ratio, over_18, subreddit)
            loss = loss_fn(preds.squeeze(), labels)
            
            total_loss += loss.item()
            predictions.extend(preds.squeeze().cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate additional metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return avg_loss, mse, r2, predictions, targets

# Training loop
EPOCHS = 5
best_val_loss = float('inf')
patience_counter = 0
max_patience = 3

print(f"\nStarting LoRA training for {EPOCHS} epochs...")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        upvote_ratio = batch['upvote_ratio'].to(device)
        over_18 = batch['over_18'].to(device)
        subreddit = batch['subreddit'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        preds = model(input_ids, attention_mask, upvote_ratio, over_18, subreddit)
        loss = loss_fn(preds.squeeze(), labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    # Validation
    val_loss, val_mse, val_r2, val_preds, val_targets = validate_model(
        model, test_loader, loss_fn, device
    )
    
    # Update learning rate
    scheduler.step(val_loss)
    
    avg_train_loss = total_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Training Loss: {avg_train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation MSE: {val_mse:.4f}")
    print(f"  Validation R²: {val_r2:.4f}")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print(f"  ✅ New best validation loss! Saving model...")
        
        # Create directory
        os.makedirs("trained_models_lora", exist_ok=True)
        
        # Save LoRA adapters only (much smaller)
        model.lora_roberta.save_pretrained("trained_models_lora/lora_adapters")
        
        # Save additional model components
        torch.save({
            'subreddit_embedding': model.subreddit_embedding.state_dict(),
            'numerical_proj': model.numerical_proj.state_dict(),
            'fusion_layer': model.fusion_layer.state_dict(),
            'regressor': model.regressor.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_r2': val_r2
        }, "trained_models_lora/additional_layers.pt")
        
        # Save preprocessing objects
        joblib.dump(score_scaler, "trained_models_lora/score_scaler.save")
        joblib.dump(subreddit_encoder, "trained_models_lora/subreddit_encoder.save")
        
        # Save tokenizer
        tokenizer.save_pretrained("trained_models_lora/tokenizer")
        
        # Save config for easy loading
        config = {
            'num_subreddits': num_subreddits,
            'embedding_dim': 32,
            'lora_config': LORA_CONFIG,
            'model_name': 'roberta-base'
        }
        joblib.dump(config, "trained_models_lora/model_config.save")
        
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"  Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    print("-" * 40)

print(f"\nLoRA Training complete! ૮₍ ´ ꒳ `₎ა")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"\nSaved files:")
print(f"  - trained_models_lora/lora_adapters/ (LoRA weights)")
print(f"  - trained_models_lora/additional_layers.pt (Other layers)")
print(f"  - trained_models_lora/score_scaler.save")
print(f"  - trained_models_lora/subreddit_encoder.save")
print(f"  - trained_models_lora/tokenizer/")
print(f"  - trained_models_lora/model_config.save")

print(f"\nSubreddit Statistics:")
print(f"  Total unique subreddits: {len(subreddit_encoder.classes_)}")
print(f"  First 10: {list(subreddit_encoder.classes_[:10])}")

print(f"\nModel Efficiency:")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Training efficiency: {100*trainable_params/total_params:.2f}% parameters trained")

print("\n🎉 LoRA training completed successfully!")