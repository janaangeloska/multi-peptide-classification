"""
ProtT5 Multilabel Training Script for GPU Cluster
Predicts 4 peptide functions: antimicrobial, antiviral, antifungal, anticancer
Datasets: dbAMP, DRAMP (80/10/10 train/val/test splits)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from tqdm.auto import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import gc
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Device & reproducibility ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ── Paths (Singularity-aware) ─────────────────────────────────────────────────
if os.path.exists('/mnt/code'):
    BASE_DIR = '/mnt'
    print("Running inside Singularity container")
else:
    BASE_DIR = '..'
    print("Running on local system")

DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Label definition ──────────────────────────────────────────────────────────
LABEL_COLS = ['antimicrobial', 'antiviral', 'antifungal', 'anticancer']
N_LABELS = len(LABEL_COLS)

print("\n" + "=" * 80)
print("PROTT5 MULTILABEL TRAINING PIPELINE")
print("=" * 80)
print(f"Start time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Labels     : {LABEL_COLS}")
print(f"Data dir   : {DATA_DIR}")
print(f"Results dir: {RESULTS_DIR}")
print("=" * 80 + "\n")

# ============================================================================
# DATA LOADING
# ============================================================================

print("Loading datasets...")
dbamp_train = pd.read_csv(os.path.join(DATA_DIR, 'dbamp_train.csv'))
dbamp_val = pd.read_csv(os.path.join(DATA_DIR, 'dbamp_val.csv'))
dbamp_test = pd.read_csv(os.path.join(DATA_DIR, 'dbamp_test.csv'))

dramp_train = pd.read_csv(os.path.join(DATA_DIR, 'dramp_train.csv'))
dramp_val = pd.read_csv(os.path.join(DATA_DIR, 'dramp_val.csv'))
dramp_test = pd.read_csv(os.path.join(DATA_DIR, 'dramp_test.csv'))

print("=== Dataset Sizes ===")
print(f"dbAMP - Train: {len(dbamp_train)}, Val: {len(dbamp_val)}, Test: {len(dbamp_test)}")
print(f"DRAMP - Train: {len(dramp_train)}, Val: {len(dramp_val)}, Test: {len(dramp_test)}")
print()


# ============================================================================
# DATASET & MODEL
# ============================================================================

def prepare_sequence_for_prott5(sequence):
    """
    ProtT5 requires:
      1. Rare amino acids (U, Z, O, B) replaced with X
      2. Spaces between every amino acid
      3. A leading <AA2fold> prefix token
    """
    sequence = (sequence
                .replace('U', 'X')
                .replace('Z', 'X')
                .replace('O', 'X')
                .replace('B', 'X'))
    return f"<AA2fold> {' '.join(list(sequence))}"


class ProtT5MultilabelDataset(Dataset):
    """PyTorch Dataset for ProtT5 multilabel classification."""

    def __init__(self, dataframe, tokenizer, label_cols=LABEL_COLS, max_length=512):
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe[label_cols].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        formatted_seq = prepare_sequence_for_prott5(self.sequences[idx])
        encoding = self.tokenizer(
            formatted_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


class ProtT5MultilabelClassifier(nn.Module):
    """ProtT5 encoder-only with a multilabel classification head.
    Uses attention-mask-weighted mean pooling over token embeddings.
    """

    def __init__(self, model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
                 n_labels=N_LABELS, dropout=0.3, freeze_t5=False):
        super().__init__()
        print(f"Loading ProtT5 model: {model_name}")
        print("Note: large model (~11 GB), may take a few minutes to download...")

        self.t5 = T5EncoderModel.from_pretrained(model_name)
        hidden_size = self.t5.config.d_model
        print(f"Hidden size: {hidden_size}")

        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False
            print("T5 parameters frozen")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Attention-mask-weighted mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask  # (batch, hidden)

        return self.classifier(self.dropout(mean_embedding))


# ============================================================================
# TRAINING & EVALUATION HELPERS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    all_logits, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

        if len(all_logits) % 5 == 0:
            torch.cuda.empty_cache()

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    all_preds = (torch.sigmoid(all_logits).numpy() >= 0.5).astype(int)

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, macro_f1


def evaluate(model, dataloader, criterion, device, label_cols=LABEL_COLS):
    model.eval()
    total_loss = 0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_probs = torch.sigmoid(all_logits).numpy()
    all_preds = (all_probs >= 0.5).astype(int)
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(dataloader)
    subset_acc = accuracy_score(all_labels, all_preds)
    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    per_label_auc = {}
    for i, label in enumerate(label_cols):
        if len(np.unique(all_labels[:, i])) > 1:
            per_label_auc[label] = roc_auc_score(all_labels[:, i], all_probs[:, i])
        else:
            per_label_auc[label] = float('nan')
    macro_auc = np.nanmean(list(per_label_auc.values()))

    return {
        'loss': avg_loss,
        'subset_accuracy': subset_acc,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'macro_auc': macro_auc,
        'per_label_auc': per_label_auc,
    }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_prott5_multilabel(train_df, val_df, test_df, dataset_name,
                            model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
                            batch_size=2, epochs=5, learning_rate=2e-5,
                            freeze_t5=False, use_scheduler=True):
    print(f"\n{'=' * 60}")
    print(f"Training ProtT5 Multilabel: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Model        : {model_name}")
    print(f"Batch size   : {batch_size}")
    print(f"Epochs       : {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Freeze T5    : {freeze_t5}")
    print(f"Scheduler    : {use_scheduler}")

    torch.cuda.empty_cache()
    gc.collect()

    print("\nLoading ProtT5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

    print("Creating datasets...")
    train_dataset = ProtT5MultilabelDataset(train_df, tokenizer, max_length=256)
    val_dataset = ProtT5MultilabelDataset(val_df, tokenizer, max_length=256)
    test_dataset = ProtT5MultilabelDataset(test_df, tokenizer, max_length=256)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=(num_workers > 0))

    print("\nInitialising ProtT5 multilabel classifier...")
    model = ProtT5MultilabelClassifier(model_name=model_name, n_labels=N_LABELS,
                                       dropout=0.3, freeze_t5=freeze_t5)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )

    scaler = torch.cuda.amp.GradScaler()

    history = {
        'train_loss': [], 'train_macro_f1': [],
        'val_loss': [], 'val_macro_f1': [], 'val_macro_auc': []
    }

    best_val_f1 = 0
    best_model_state = None
    patience = 3
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_macro_f1'].append(train_f1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_macro_auc'].append(val_metrics['macro_auc'])

        print(f"\nTrain  - Loss: {train_loss:.4f}  Macro F1: {train_f1:.4f}")
        print(f"Val    - Loss: {val_metrics['loss']:.4f}  "
              f"Macro F1: {val_metrics['macro_f1']:.4f}  "
              f"Macro AUC: {val_metrics['macro_auc']:.4f}")
        print(f"         Subset Acc: {val_metrics['subset_accuracy']:.4f}")
        print("         Per-label AUC:", val_metrics['per_label_auc'])

        if scheduler is not None:
            scheduler.step(val_metrics['macro_f1'])
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ New best model! (Macro F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping!")
                break

        torch.cuda.empty_cache()

    print("\nLoading best model for testing...")
    model.load_state_dict(best_model_state)

    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print(f"{'=' * 60}")
    for k, v in test_metrics.items():
        if k == 'per_label_auc':
            print("  Per-label AUC:")
            for label, auc in v.items():
                print(f"    {label:15s}: {auc:.4f}")
        else:
            print(f"  {k:20s}: {v:.4f}")

    return model, history, test_metrics


def plot_training_history(history, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch');
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{dataset_name} - Loss')
    axes[0].legend();
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_macro_f1'], 'b-', label='Train Macro F1')
    axes[1].plot(epochs, history['val_macro_f1'], 'r-', label='Val Macro F1')
    axes[1].plot(epochs, history['val_macro_auc'], 'g-', label='Val Macro AUC')
    axes[1].set_xlabel('Epoch');
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'{dataset_name} - Metrics')
    axes[1].legend();
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'prott5_multilabel_{dataset_name.lower()}_training.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training plot: {save_path}")
    plt.close()


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":

    # ── dbAMP ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING PROTT5 ON dbAMP DATASET")
    print("=" * 80 + "\n")

    dbamp_model, dbamp_history, dbamp_results = train_prott5_multilabel(
        dbamp_train, dbamp_val, dbamp_test,
        dataset_name="dbAMP",
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
        batch_size=4,
        epochs=10,
        learning_rate=2e-5,
        freeze_t5=True,  # frozen backbone for faster first run
        use_scheduler=False
    )

    plot_training_history(dbamp_history, "dbAMP")

    dbamp_model_path = os.path.join(RESULTS_DIR, 'prott5_multilabel_dbamp_final.pth')
    torch.save(dbamp_model.state_dict(), dbamp_model_path)
    print(f"✓ dbAMP model saved: {dbamp_model_path}")

    del dbamp_model
    torch.cuda.empty_cache()
    gc.collect()

    # ── DRAMP ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING PROTT5 ON DRAMP DATASET")
    print("=" * 80 + "\n")

    dramp_model, dramp_history, dramp_results = train_prott5_multilabel(
        dramp_train, dramp_val, dramp_test,
        dataset_name="DRAMP",
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
        batch_size=2,  # unfrozen needs more memory
        epochs=20,
        learning_rate=1e-5,
        freeze_t5=False,  # full fine-tuning
        use_scheduler=True
    )

    plot_training_history(dramp_history, "DRAMP")

    dramp_model_path = os.path.join(RESULTS_DIR, 'prott5_multilabel_dramp_final.pth')
    torch.save(dramp_model.state_dict(), dramp_model_path)
    print(f"✓ DRAMP model saved: {dramp_model_path}")

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAVING FINAL RESULTS")
    print("=" * 80 + "\n")

    rows = []
    for dataset_name, res in [('dbAMP', dbamp_results), ('DRAMP', dramp_results)]:
        row = {
            'Model': 'ProtT5',
            'Dataset': dataset_name,
            'Subset_Accuracy': res['subset_accuracy'],
            'Macro_Precision': res['macro_precision'],
            'Macro_Recall': res['macro_recall'],
            'Macro_F1': res['macro_f1'],
            'Macro_AUC': res['macro_auc'],
        }
        for label, auc in res['per_label_auc'].items():
            row[f'AUC_{label}'] = auc
        rows.append(row)

    results_path = os.path.join(RESULTS_DIR, 'results_prott5_multilabel.csv')
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Results saved: {results_path}")

    # ── Comparison bar chart ──────────────────────────────────────────────────
    metrics_to_plot = ['macro_precision', 'macro_recall', 'macro_f1', 'macro_auc']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 4))

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [dbamp_results[metric], dramp_results[metric]]
        ax.bar(['dbAMP', 'DRAMP'], vals, color=['#3498db', '#e74c3c'])
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim([0, 1]);
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    plt.suptitle('ProtT5 Multilabel Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    comparison_path = os.path.join(RESULTS_DIR, 'prott5_multilabel_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {comparison_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
