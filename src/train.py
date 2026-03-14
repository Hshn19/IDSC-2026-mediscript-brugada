import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.dataset import BrugadaDataset
from src.preprocessing import preprocess_signal


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Class Weight ──────────────────────────────────────────────────────────────

def get_pos_weight(y_train):
    """
    Compute positive class weight for BCEWithLogitsLoss.

    With 195 Normal and 58 Brugada in training set:
    pos_weight = 195 / 58 = 3.36

    Effect: the loss penalizes missing a Brugada case 3.36x more than
    missing a Normal case — directly addresses class imbalance without
    needing SMOTE for the DL model.
    """
    n_neg = sum(1 for y in y_train if y == 0)
    n_pos = sum(1 for y in y_train if y == 1)
    weight = n_neg / n_pos
    print(f"Positive class weight: {weight:.4f}  ({n_neg} Normal / {n_pos} Brugada)")
    return torch.tensor([weight], dtype=torch.float32)


# ── Training & Evaluation Loops ───────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss  = 0.0
    all_probs   = []
    all_labels  = []

    for signals, labels in tqdm(loader, desc="  Train", leave=False):
        signals = signals.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals).squeeze(1)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients on small batches
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0   # only one class seen this epoch — too few samples
    return avg_loss, auroc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for signals, labels in loader:
        signals = signals.to(device)
        labels  = labels.to(device)

        logits = model(signals).squeeze(1)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0
    return avg_loss, auroc, np.array(all_probs), np.array(all_labels)


# ── Main Training Function ────────────────────────────────────────────────────

def run_training(model, train_split, val_split, data_dir, config):
    """
    Full training loop with:
    - Class-weighted BCE loss
    - AdamW optimizer
    - Cosine annealing LR schedule
    - Early stopping on val AUROC
    - Best model checkpoint saving

    Args:
        model        : BrugadaCNN instance
        train_split  : dict with 'ids' and 'labels'
        val_split    : dict with 'ids' and 'labels'
        data_dir     : path to ECG files folder
        config       : dict of hyperparameters

    Returns:
        history : dict of loss and AUROC curves
    """
    seed_everything(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    model  = model.to(device)

    # Datasets — preprocessing applied via transform
    train_dataset = BrugadaDataset(
        train_split['ids'], train_split['labels'],
        data_dir, transform=preprocess_signal
    )
    val_dataset = BrugadaDataset(
        val_split['ids'], val_split['labels'],
        data_dir, transform=preprocess_signal
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=0, pin_memory=False)

    # Loss, optimizer, scheduler
    pos_weight = get_pos_weight(train_split['labels']).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=config['lr'], weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['epochs'])

    # Training state
    best_auroc   = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history      = {
        'train_loss': [], 'val_loss': [],
        'train_auroc': [], 'val_auroc': []
    }

    os.makedirs('outputs/checkpoints', exist_ok=True)
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} {'Train AUROC':>11} "
          f"| {'Val Loss':>8} {'Val AUROC':>9} | {'LR':>8}")
    print("-" * 70)

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_auroc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auroc'].append(train_auroc)
        history['val_auroc'].append(val_auroc)

        # Checkpoint on best val AUROC
        marker = ""
        if val_auroc > best_auroc:
            best_auroc   = val_auroc
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pt')
            marker = "  ✓ best"
        else:
            patience_ctr += 1

        print(f"{epoch:>6} | {train_loss:>10.4f} {train_auroc:>11.4f} "
              f"| {val_loss:>8.4f} {val_auroc:>9.4f} "
              f"| {current_lr:>8.6f}{marker}")

        # Early stopping
        if patience_ctr >= config['patience']:
            print(f"\n⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {config['patience']} epochs)")
            break

    print(f"\n🏆 Best Val AUROC: {best_auroc:.4f} at epoch {best_epoch}")
    print(f"   Checkpoint saved: outputs/checkpoints/best_model.pt")

    # Save history
    with open('outputs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return history