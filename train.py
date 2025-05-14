import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Example simple classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom Dataset
class WebQSPDataset(Dataset):
    def __init__(self, q_embs, labels):
        self.q_embs = q_embs
        self.labels = labels
    def __len__(self):
        return len(self.q_embs)
    def __getitem__(self, idx):
        return self.q_embs[idx], self.labels[idx]

def load_labels(indices, dataset):
    labels = []
    for i in indices:
        ans = dataset.iloc[i]['answer']
        # Handle numpy array case
        if isinstance(ans, np.ndarray):
            ans = '|'.join(ans.astype(str))  # Convert array elements to strings and join
        # Handle list case
        elif isinstance(ans, list):
            ans = '|'.join(map(str, ans))  # Convert list elements to strings and join
        # Handle other cases (should be string)
        else:
            ans = str(ans)
        label = 1 if 'support' in ans.lower() else 0
        labels.append(label)
    return torch.tensor(labels, dtype=torch.long)

def main():
    # --- Load embeddings and dataset ---
    path = 'dataset/webqsp'
    q_embs = torch.load(f'{path}/q_embs.pt')  # shape: [N, D]
    # Load your dataset as before (e.g. with load_parquet_dataset)
    from src.dataset.preprocess.webqsp import load_parquet_dataset
    data_dir = '/content/g-retriever/RoG-webqsp/data'
    dataset_dict = load_parquet_dataset(data_dir)
    dataset = pd.concat([dataset_dict['train'].to_pandas(), 
                         dataset_dict['validation'].to_pandas(), 
                         dataset_dict['test'].to_pandas()], ignore_index=True)

    # --- Load split indices ---
    with open(f'{path}/split/train_indices.txt') as f:
        train_idx = [int(x) for x in f]
    with open(f'{path}/split/val_indices.txt') as f:
        val_idx = [int(x) for x in f]
    with open(f'{path}/split/test_indices.txt') as f:
        test_idx = [int(x) for x in f]

    # --- Prepare datasets ---
    train_labels = load_labels(train_idx, dataset)
    val_labels = load_labels(val_idx, dataset)
    test_labels = load_labels(test_idx, dataset)

    train_set = WebQSPDataset(q_embs[train_idx], train_labels)
    val_set = WebQSPDataset(q_embs[val_idx], val_labels)
    test_set = WebQSPDataset(q_embs[test_idx], test_labels)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    # --- Model, Loss, Optimizer ---
    input_dim = q_embs.shape[1]
    model = SimpleClassifier(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Training Loop ---
    best_val_acc = 0
    for epoch in range(10):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Val Acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{path}/best_model.pt")
            print("Saved new best model.")
            # --- Also save to Google Drive for persistence ---
            drive_path = '/content/drive/MyDrive/Projets TPs GL4/PFA GL4/webqsp'
            drive_model_dir = os.path.join(drive_path, path)
            os.makedirs(drive_model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(drive_model_dir, "best_model.pt"))

    # --- Test ---
    model.load_state_dict(torch.load(f"{path}/best_model.pt"))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    # --- Save test results to Google Drive ---
    test_results_path = f"{path}/test_results.txt"
    with open(test_results_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
    drive_results_dir = os.path.join(drive_path, path)
    os.makedirs(drive_results_dir, exist_ok=True)
    with open(os.path.join(drive_results_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

from sentence_transformers import SentenceTransformer

def test_with_prompt(model_path, input_dim):
    # Load the trained classifier
    model = SimpleClassifier(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the same embedding model used for q_embs
    embed_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

    while True:
        print("\nEnter a question to classify (or 'quit' to exit):")
        user_input = input().strip()
        if user_input.lower() == 'quit':
            break

        # Generate embedding for the input question
        question_embedding = embed_model.encode([user_input], convert_to_tensor=True)
        # Make prediction
        with torch.no_grad():
            logits = model(question_embedding)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax().item()
            confidence = probs[0][pred_class].item()

        class_name = "SUPPORT" if pred_class == 1 else "NOT SUPPORT"
        print(f"\nPrediction: {class_name} (confidence: {confidence:.2%})")

if __name__ == "__main__":
    # main()
    path = 'dataset/webqsp'
    q_embs = torch.load(f'{path}/q_embs.pt')
    input_dim = q_embs.shape[1]
    model_path = f'{path}/best_model.pt'
    test_with_prompt(model_path, input_dim)