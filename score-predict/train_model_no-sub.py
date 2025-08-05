import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import RobertaModel
from tqdm import tqdm
from data_tokenizer import prepare_data

print("Preparing data... \n ૮₍ ´ ꒳ `₎ა \n")

# Enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
tokenizer, tokenized, upvote_ratios, over_18_flags, normalized_scores, score_scaler = prepare_data()

# Combine features
input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']
numerical_features = torch.cat([upvote_ratios, over_18_flags], dim=1)

# Dataset & Split
dataset = TensorDataset(input_ids, attention_mask, numerical_features, normalized_scores)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

# Model
class RobertaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta.train()  # Unfreeze
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

model = RobertaRegressor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

# Training loop
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for input_ids, attention_mask, num_feats, targets in loop:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        num_feats, targets = num_feats.to(device), targets.to(device)

        preds = model(input_ids, attention_mask, num_feats)
        loss = loss_fn(preds.squeeze(), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} | Training Loss: {total_loss / len(train_loader):.4f}")

# Save model and scaler
torch.save(model.state_dict(), "roberta_score_regressor.pt")
import joblib
joblib.dump(score_scaler, "score_scaler.save")

print("Training complete. Model and scaler saved. \n ૮₍ ´ ꒳ `₎ა ")