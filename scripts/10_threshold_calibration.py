import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_auc_score, hamming_loss,
    multilabel_confusion_matrix
)
from transformers import (
    BertTokenizer, BertModel,
    AutoTokenizer, EsmModel,
    T5Tokenizer, T5EncoderModel
)
import warnings
import os
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

PROTBERT_DBAMP_PTH = '../results/protbert_multilabel_dbamp_final2.pth'
PROTBERT_DRAMP_PTH = '../results/protbert_multilabel_dramp_final2.pth'
ESM2_DBAMP_PTH = '../results/esm2_multilabel_dbamp_final2.pth'
ESM2_DRAMP_PTH = '../results/esm2_multilabel_dramp_final2.pth'
PROTT5_DBAMP_PTH = '../results/prott5_multilabel_dbamp_final2.pth'
PROTT5_DRAMP_PTH = '../results/prott5_multilabel_dramp_final2.pth'

DBAMP_TEST_CSV = '../data/dbamp_test.csv'
DRAMP_TEST_CSV = '../data/dramp_test.csv'
DBAMP_VAL_CSV  = '../data/dbamp_val.csv'
DRAMP_VAL_CSV  = '../data/dramp_val.csv'

USE_SAVED_WEIGHTS = True
OUTPUT_DIR = '../results/threshold_calibration'
BATCH_SIZE = 64
THRESHOLD = 0.5  # default, will be overridden per label after calibration

THRESHOLD_GRID = np.arange(0.1, 0.91, 0.05).round(2)  # 0.10 … 0.90

# Label definition — must match training scripts
LABEL_COLS = ['antimicrobial', 'antiviral', 'antifungal', 'anticancer']
N_LABELS = len(LABEL_COLS)

# ── IEEE Color Palette ──────────────────────────────────────────────────────
# Based on standard IEEE publication color guidelines:
#   #0072BD  IEEE blue       #D95319  orange-red
#   #EDB120  gold            #7E2F8E  purple
#   #77AC30  green           #4DBEEE  light blue
#   #A2142F  dark red        #808080  neutral gray

LABEL_COLORS = {
    'antimicrobial': '#0072BD',  # IEEE blue
    'antiviral': '#D95319',  # orange-red
    'antifungal': '#77AC30',  # green
    'anticancer': '#7E2F8E',  # purple
}

# Error-type colours (TP/TN/FP/FN)
_ERROR_COLORS = {
    'TP': '#77AC30',  # green   – correct positive
    'TN': '#0072BD',  # blue    – correct negative
    'FP': '#EDB120',  # gold    – false alarm
    'FN': '#D95319',  # orange-red – missed positive
}

# Dataset bar colours
_DS_COLORS = {
    'dbAMP': '#0072BD',  # IEEE blue
    'DRAMP': '#D95319',  # orange-red
}

# Model colours for grouped bar charts
_MODEL_COLORS = {
    'ESM-2': '#0072BD',
    'ProtBERT': '#D95319',
    'ProtT5': '#EDB120',
}

# Global matplotlib style – clean, publication-ready
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.color': '#CCCCCC',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#AAAAAA',
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0,
}
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0,
}


def seq_hydrophobicity(seq):
    vals = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq.upper()]
    return np.mean(vals) if vals else 0.0


def seq_charge(seq):
    return sum(CHARGE.get(aa, 0) for aa in seq.upper())


def seq_length(seq):
    return len(seq)


class ProtBERTMultilabelClassifier(nn.Module):
    def __init__(self, n_labels=N_LABELS, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class ESM2MultilabelClassifier(nn.Module):
    def __init__(self, model_name='facebook/esm2_t6_8M_UR50D',
                 n_labels=N_LABELS, dropout=0.3):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.esm.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class ProtT5MultilabelClassifier(nn.Module):
    def __init__(self, model_name='Rostlab/prot_t5_xl_half_uniref50-enc',
                 n_labels=N_LABELS, dropout=0.3, freeze_t5=False):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.t5.config.d_model, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        mean_emb = torch.sum(embeddings * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        return self.classifier(self.dropout(mean_emb))


def load_model(model_obj, pth_path, label):
    if USE_SAVED_WEIGHTS and os.path.exists(pth_path):
        model_obj.load_state_dict(torch.load(pth_path, map_location=device))
        print(f'  Weights: {pth_path}')
    else:
        print(f'  Warning {pth_path} does not exist - overtrained weights')
    model_obj.eval().to(device)
    print(f'  {label} ready.')
    return model_obj


def format_sequence(sequence, model_type):
    if model_type == 'bert':
        return ' '.join(list(sequence))
    elif model_type == 't5':
        seq = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
        return f"<AA2fold> {' '.join(list(seq))}"
    else:
        return sequence


def run_inference(df, model, tokenizer, model_type):
    all_probs = []
    sequences = df['sequence'].tolist()
    n = len(sequences)

    for start in range(0, n, BATCH_SIZE):
        batch = sequences[start: start + BATCH_SIZE]
        formatted = [format_sequence(s, model_type) for s in batch]

        encoding = tokenizer(
            formatted,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())

        if (start // BATCH_SIZE + 1) % 5 == 0:
            print(f'    {min(start + BATCH_SIZE, n)}/{n} done...')

    pred_probs = np.vstack(all_probs)
    pred_binary = (pred_probs >= THRESHOLD).astype(int)
    return pred_probs, pred_binary


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def build_error_df(df, true_labels, pred_binary, pred_probs,
                   model_name, dataset_name, label_cols=LABEL_COLS):
    records = []
    for i, seq in enumerate(df['sequence']):
        rec = {
            'model': model_name,
            'dataset': dataset_name,
            'sequence': seq,
            'length': seq_length(seq),
            'charge': seq_charge(seq),
            'hydrophobicity': seq_hydrophobicity(seq),
        }
        n_errors = 0
        for j, lbl in enumerate(label_cols):
            t = int(true_labels[i, j])
            p = int(pred_binary[i, j])
            prob = round(float(pred_probs[i, j]), 4)

            if t == 1 and p == 1:
                etype = 'TP'
            elif t == 0 and p == 0:
                etype = 'TN'
            elif t == 0 and p == 1:
                etype = 'FP'
            else:
                etype = 'FN'  # missed positive

            rec[f'true_{lbl}'] = t
            rec[f'pred_{lbl}'] = p
            rec[f'prob_{lbl}'] = prob
            rec[f'etype_{lbl}'] = etype
            if etype in ('FP', 'FN'):
                n_errors += 1

        rec['n_label_errors'] = n_errors
        rec['any_error'] = n_errors > 0
        rec['exact_match'] = np.array_equal(
            true_labels[i], pred_binary[i])
        records.append(rec)

    return pd.DataFrame(records)


def compute_per_label_metrics(true_labels, pred_binary, pred_probs,
                              label_cols=LABEL_COLS):
    metrics = {}

    metrics['subset_accuracy'] = np.all(
        true_labels == pred_binary, axis=1).mean()
    metrics['hamming_loss'] = hamming_loss(true_labels, pred_binary)
    metrics['macro_f1'] = f1_score(true_labels, pred_binary,
                                   average='macro', zero_division=0)
    metrics['micro_f1'] = f1_score(true_labels, pred_binary,
                                   average='micro', zero_division=0)
    try:
        metrics['macro_auc'] = roc_auc_score(
            true_labels, pred_probs, average='macro')
    except ValueError:
        metrics['macro_auc'] = None

    mcm = multilabel_confusion_matrix(true_labels, pred_binary)
    for i, lbl in enumerate(label_cols):
        tn, fp, fn, tp = mcm[i].ravel()
        metrics[f'{lbl}_tp'] = int(tp)
        metrics[f'{lbl}_tn'] = int(tn)
        metrics[f'{lbl}_fp'] = int(fp)
        metrics[f'{lbl}_fn'] = int(fn)
        metrics[f'{lbl}_fn_rate'] = round(fn / (fn + tp + 1e-9), 4)
        metrics[f'{lbl}_fp_rate'] = round(fp / (fp + tn + 1e-9), 4)
        metrics[f'{lbl}_f1'] = round(
            f1_score(true_labels[:, i], pred_binary[:, i], zero_division=0), 4)
        try:
            metrics[f'{lbl}_auc'] = round(
                roc_auc_score(true_labels[:, i], pred_probs[:, i]), 4)
        except ValueError:
            metrics[f'{lbl}_auc'] = None

    return metrics


def plot_multilabel_confusion_matrices(true_labels, pred_binary, model_name,
                                       dataset_name, output_dir,
                                       label_cols=LABEL_COLS):
    n = len(label_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.suptitle(f'{model_name} — {dataset_name}\nPer-label Confusion Matrices',
                 fontsize=11, fontweight='bold')

    # IEEE-style blue colormap for confusion matrix cells
    mcm = multilabel_confusion_matrix(true_labels, pred_binary)
    for ax, (i, lbl) in zip(axes, enumerate(label_cols)):
        cm = mcm[i]
        tn, fp, fn, tp = cm.ravel()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'],
                    linewidths=0.5, linecolor='#AAAAAA')
        fn_r = fn / (fn + tp + 1e-9)
        fp_r = fp / (fp + tn + 1e-9)
        ax.set_title(f'{lbl}\nFN rate={fn_r:.2%}  FP rate={fp_r:.2%}',
                     fontsize=9, fontweight='bold',
                     color=LABEL_COLORS.get(lbl, 'black'))
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)

        # Annotate FN/FP cells with IEEE error colours
        ax.text(0.5, 1.5, 'FN', ha='center', va='center', fontsize=9,
                color=_ERROR_COLORS['FN'], fontweight='bold')
        ax.text(1.5, 0.5, 'FP', ha='center', va='center', fontsize=9,
                color=_ERROR_COLORS['FP'], fontweight='bold')

    plt.tight_layout()
    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'cm_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_error_profiles(error_df, model_name, dataset_name, output_dir,
                        label_cols=LABEL_COLS):
    props = [
        ('length', 'Sequence length (AA)'),
        ('charge', 'Total charge'),
        ('hydrophobicity', 'Average hydrophobicity (KD)'),
    ]

    for lbl in label_cols:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f'{model_name} — {dataset_name} — [{lbl}]\n'
            f'Error profile by physicochemical properties',
            fontsize=11, fontweight='bold'
        )
        for ax, (col, xlabel) in zip(axes, props):
            for etype, color in _ERROR_COLORS.items():
                subset = error_df[error_df[f'etype_{lbl}'] == etype][col]
                if len(subset) == 0:
                    continue
                ax.hist(subset, bins=20, alpha=0.55, color=color,
                        label=f'{etype} (n={len(subset)})', density=True,
                        edgecolor='white', linewidth=0.5)
                ax.axvline(subset.mean(), color=color, linewidth=1.8,
                           linestyle='--', alpha=0.85)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(xlabel, fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)

        plt.tight_layout()
        tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}_{lbl}'
        path = os.path.join(output_dir, f'error_profiles_{tag}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_confidence_errors(error_df, model_name, dataset_name, output_dir,
                           label_cols=LABEL_COLS):
    n = len(label_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle(f'{model_name} — {dataset_name}\nConfidence vs length',
                 fontsize=11, fontweight='bold')

    for ax, lbl in zip(axes, label_cols):
        for etype, color in _ERROR_COLORS.items():
            sub = error_df[error_df[f'etype_{lbl}'] == etype]
            if len(sub) == 0:
                continue
            ax.scatter(sub['length'], sub[f'prob_{lbl}'],
                       c=color, alpha=0.55, s=18,
                       label=f'{etype} (n={len(sub)})')

        ax.axhline(THRESHOLD, color='#808080', linestyle='--', linewidth=1.0,
                   alpha=0.8, label=f'Threshold ({THRESHOLD})')
        ax.set_xlabel('Length (AA)', fontsize=10)
        ax.set_ylabel(f'P({lbl})', fontsize=10)
        ax.set_title(lbl, fontsize=10, fontweight='bold',
                     color=LABEL_COLORS.get(lbl, 'black'))
        ax.legend(fontsize=7)

    plt.tight_layout()
    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'confidence_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_fn_rate_comparison(summary_rows, output_dir, label_cols=LABEL_COLS):
    for ds_name in set(r['dataset'] for r in summary_rows):
        subset = [r for r in summary_rows if r['dataset'] == ds_name]
        models = [r['model'] for r in subset]

        fn_matrix = np.array([
            [r[f'{lbl}_fn_rate'] for lbl in label_cols]
            for r in subset
        ])

        fig, ax = plt.subplots(figsize=(len(label_cols) * 2, len(models) * 1.4 + 1))
        # 'Reds' aligns with IEEE convention: higher FN = warmer (worse)
        sns.heatmap(fn_matrix, annot=True, fmt='.2%', cmap='Reds',
                    xticklabels=label_cols, yticklabels=models,
                    linewidths=0.5, linecolor='#AAAAAA',
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(f'FN Rate per model and label [{ds_name}]',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Label', fontsize=10)
        ax.set_ylabel('Model', fontsize=10)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f'fn_rate_heatmap_{ds_name.lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_per_label_f1_comparison(summary_rows, output_dir, label_cols=LABEL_COLS):
    models = list(dict.fromkeys(r['model'] for r in summary_rows))
    datasets = list(dict.fromkeys(r['dataset'] for r in summary_rows))

    for ds_name in datasets:
        subset = {r['model']: r for r in summary_rows if r['dataset'] == ds_name}
        x = np.arange(len(label_cols))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        for offset, mname in zip(
                np.linspace(-width, width, len(models)), models):
            if mname not in subset:
                continue
            vals = [subset[mname].get(f'{lbl}_f1', 0.0) or 0.0
                    for lbl in label_cols]
            bars = ax.bar(x + offset, vals, width,
                          label=mname,
                          color=_MODEL_COLORS.get(mname, '#808080'),
                          alpha=0.85, edgecolor='white', linewidth=0.8)
            ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

        ax.set_title(f'Per-label F1 per model [{ds_name}]',
                     fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(label_cols, fontsize=10)
        ax.set_ylabel('F1 Score')
        ax.set_ylim([0, 1.12])
        ax.legend(title='Model', fontsize=9)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f'per_label_f1_{ds_name.lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')

def plot_per_label_auc_comparison(summary_rows, output_dir, label_cols=LABEL_COLS):
    """
    Mirror of plot_per_label_f1_comparison but for per-label AUC-ROC.
    Skips labels where AUC is None (undefined) for all models on a dataset.
    """
    models = list(dict.fromkeys(r['model'] for r in summary_rows))
    datasets = list(dict.fromkeys(r['dataset'] for r in summary_rows))

    for ds_name in datasets:
        subset = {r['model']: r for r in summary_rows if r['dataset'] == ds_name}
        x = np.arange(len(label_cols))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        for offset, mname in zip(
                np.linspace(-width, width, len(models)), models):
            if mname not in subset:
                continue
            vals = []
            for lbl in label_cols:
                v = subset[mname].get(f'{lbl}_auc', None)
                vals.append(float(v) if v is not None else 0.0)

            bars = ax.bar(x + offset, vals, width,
                          label=mname,
                          color=_MODEL_COLORS.get(mname, '#808080'),
                          alpha=0.85, edgecolor='white', linewidth=0.8)
            ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

        ax.set_title(f'Per-label AUC-ROC per model [{ds_name}]',
                     fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(label_cols, fontsize=10)
        ax.set_ylabel('AUC-ROC')
        ax.set_ylim([0, 1.12])

        # Reference line at 0.5 (random baseline)
        ax.axhline(0.5, color='#808080', linestyle='--',
                   linewidth=0.9, alpha=0.7, label='Random (0.5)')

        ax.legend(title='Model', fontsize=9)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f'per_label_auc_{ds_name.lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')

def plot_n_label_errors_distribution(error_df, model_name,
                                     dataset_name, output_dir):
    counts = error_df['n_label_errors'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color='#7E2F8E', alpha=0.85,
                  edgecolor='white', linewidth=0.8)
    ax.bar_label(bars, fontsize=9, padding=2)
    ax.set_xlabel('Number of wrong labels per sequence', fontsize=11)
    ax.set_ylabel('Number of sequences', fontsize=11)
    ax.set_title(f'{model_name} — {dataset_name}\n'
                 f'Distribution of errors per sequence',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(counts.index)
    plt.tight_layout()
    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'n_label_errors_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_aggregate_metrics_summary(summary_rows, output_dir):
    df = pd.DataFrame(summary_rows)
    models = df['model'].unique()
    datasets = df['dataset'].unique()

    metrics_cfg = [
        ('subset_accuracy', 'Subset Accuracy'),
        ('macro_f1', 'Macro F1'),
        ('macro_auc', 'Macro AUC'),
        ('hamming_loss', 'Hamming Loss'),
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, len(metrics_cfg),
                             figsize=(5 * len(metrics_cfg), 5))
    fig.suptitle('Aggregate Metrics per model and dataset',
                 fontsize=12, fontweight='bold')

    for ax, (metric, title) in zip(axes, metrics_cfg):
        for offset, ds in zip([-width / 2, width / 2], datasets):
            vals = []
            for m in models:
                row = df[(df['model'] == m) & (df['dataset'] == ds)]
                vals.append(float(row[metric].values[0])
                            if len(row) and row[metric].values[0] is not None
                            else 0.0)
            bars = ax.bar(x + offset, vals, width, label=ds,
                          color=_DS_COLORS.get(ds, '#808080'),
                          alpha=0.85, edgecolor='white', linewidth=0.8)
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10, ha='right')
        ax.set_ylim([0, 1.12])
        ax.legend(title='Dataset', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'aggregate_metrics_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ============================================================================
# THRESHOLD CALIBRATION
# ============================================================================

def find_optimal_thresholds(val_true, val_probs, label_cols=LABEL_COLS,
                             grid=None):
    """
    За секоја класа одделно, пребарува праг кој го максимизира F1
    на validation сетот. Враќа речник {label: best_threshold}.
    """
    if grid is None:
        grid = THRESHOLD_GRID

    best_thresholds = {}
    for i, lbl in enumerate(label_cols):
        best_f1, best_thr = -1.0, 0.5
        for thr in grid:
            preds = (val_probs[:, i] >= thr).astype(int)
            f1 = f1_score(val_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        best_thresholds[lbl] = float(best_thr)
        print(f'    {lbl:15s}: best_thr={best_thr:.2f}  val_F1={best_f1:.4f}')
    return best_thresholds


def apply_thresholds(probs, thresholds, label_cols=LABEL_COLS):
    """Применува per-label прагови на prob матрица. Враќа бинарна матрица."""
    binary = np.zeros_like(probs, dtype=int)
    for i, lbl in enumerate(label_cols):
        binary[:, i] = (probs[:, i] >= thresholds[lbl]).astype(int)
    return binary


def plot_threshold_f1_curves(val_true, val_probs, model_name,
                              dataset_name, output_dir,
                              label_cols=LABEL_COLS, grid=None):
    """
    Црта F1 vs праг крива за секоја класа — покажува каде лежи
    оптималниот праг и зошто 0.5 може да биде лош избор.
    """
    if grid is None:
        grid = THRESHOLD_GRID

    fig, axes = plt.subplots(1, len(label_cols),
                             figsize=(5 * len(label_cols), 4))
    fig.suptitle(f'{model_name} — {dataset_name}\n'
                 f'F1 vs Threshold per label (validation set)',
                 fontsize=11, fontweight='bold')

    for ax, (i, lbl) in zip(axes, enumerate(label_cols)):
        f1s = []
        for thr in grid:
            preds = (val_probs[:, i] >= thr).astype(int)
            f1s.append(f1_score(val_true[:, i], preds, zero_division=0))

        best_idx = int(np.argmax(f1s))
        ax.plot(grid, f1s, color=LABEL_COLORS.get(lbl, '#333333'),
                linewidth=2.0, marker='o', markersize=4)
        ax.axvline(grid[best_idx], color='#A2142F', linestyle='--',
                   linewidth=1.2, label=f'Best thr={grid[best_idx]:.2f}')
        ax.axvline(0.5, color='#808080', linestyle=':', linewidth=1.0,
                   label='Default (0.5)')
        ax.set_title(lbl, fontsize=10, fontweight='bold',
                     color=LABEL_COLORS.get(lbl, 'black'))
        ax.set_xlabel('Threshold', fontsize=9)
        ax.set_ylabel('F1', fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8)

    plt.tight_layout()
    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'threshold_f1_curves_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_threshold_improvement(results_default, results_calibrated,
                                model_name, dataset_name,
                                output_dir, label_cols=LABEL_COLS):
    """
    Споредба на per-label F1 пред и по threshold calibration.
    Покажува колку се подобрува секоја класа.
    """
    x = np.arange(len(label_cols))
    width = 0.35

    f1_default    = [results_default.get(f'{lbl}_f1', 0.0) or 0.0
                     for lbl in label_cols]
    f1_calibrated = [results_calibrated.get(f'{lbl}_f1', 0.0) or 0.0
                     for lbl in label_cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, f1_default, width,
                   label='Default (thr=0.5)',
                   color='#808080', alpha=0.85,
                   edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, f1_calibrated, width,
                   label='Calibrated (per-label)',
                   color=_MODEL_COLORS.get(model_name, '#0072BD'),
                   alpha=0.85, edgecolor='white', linewidth=0.8)

    ax.bar_label(bars1, fmt='%.3f', fontsize=8, padding=2)
    ax.bar_label(bars2, fmt='%.3f', fontsize=8, padding=2)

    ax.set_title(f'{model_name} — {dataset_name}\n'
                 f'Per-label F1: Default vs Calibrated threshold',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_cols, fontsize=10)
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=9)
    plt.tight_layout()

    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'threshold_improvement_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print('=' * 70)
    print('THRESHOLD CALIBRATION — ProtBERT · ESM-2 · ProtT5  [MULTILABEL]')
    print(f'Labels: {LABEL_COLS}')
    print(f'Start : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
        print(f'VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('=' * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load datasets ────────────────────────────────────────────────────────
    val_datasets  = {}
    test_datasets = {}
    for name, val_path, test_path in [
        ('dbAMP', DBAMP_VAL_CSV,  DBAMP_TEST_CSV),
        ('DRAMP', DRAMP_VAL_CSV,  DRAMP_TEST_CSV),
    ]:
        val_datasets[name]  = pd.read_csv(val_path)
        test_datasets[name] = pd.read_csv(test_path)
        print(f'\n{name} val : {len(val_datasets[name])} sequences')
        print(f'{name} test: {len(test_datasets[name])} sequences')
        for lbl in LABEL_COLS:
            pos_v = val_datasets[name][lbl].sum()
            pos_t = test_datasets[name][lbl].sum()
            print(f'  {lbl:15s}: val={pos_v}  test={pos_t}')

    pth_map = {
        'ESM-2':    {'dbAMP': ESM2_DBAMP_PTH,    'DRAMP': ESM2_DRAMP_PTH},
        'ProtBERT': {'dbAMP': PROTBERT_DBAMP_PTH, 'DRAMP': PROTBERT_DRAMP_PTH},
        'ProtT5':   {'dbAMP': PROTT5_DBAMP_PTH,   'DRAMP': PROTT5_DRAMP_PTH},
    }

    model_configs = [
        ('ESM-2',
         lambda: ESM2MultilabelClassifier(),
         lambda: AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'),
         'esm'),
        ('ProtBERT',
         lambda: ProtBERTMultilabelClassifier(),
         lambda: BertTokenizer.from_pretrained('Rostlab/prot_bert',
                                               do_lower_case=False),
         'bert'),
        ('ProtT5',
         lambda: ProtT5MultilabelClassifier(freeze_t5=True),
         lambda: T5Tokenizer.from_pretrained(
             'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False),
         't5'),
    ]

    summary_rows = []  # default threshold results
    calib_rows   = []  # calibrated threshold results

    for model_name, model_fn, tok_fn, mtype in model_configs:
        print(f'\n{"─" * 60}')
        print(f'── {model_name}')
        print(f'{"─" * 60}')

        tokenizer = tok_fn()

        for dataset_name in ['dbAMP', 'DRAMP']:
            print(f'\n  [{dataset_name}]')
            pth = pth_map[model_name][dataset_name]
            model_obj = load_model(model_fn(), pth,
                                   f'{model_name} [{dataset_name}]')

            val_df   = val_datasets[dataset_name]
            test_df  = test_datasets[dataset_name]
            val_true  = val_df[LABEL_COLS].values.astype(int)
            test_true = test_df[LABEL_COLS].values.astype(int)

            # ── Inference ────────────────────────────────────────────────────
            print('  Running inference on validation set...')
            val_probs, _  = run_inference(val_df,  model_obj, tokenizer, mtype)
            print('  Running inference on test set...')
            test_probs, _ = run_inference(test_df, model_obj, tokenizer, mtype)

            # ── Default threshold (0.5) ───────────────────────────────────────
            test_binary_default = (test_probs >= 0.5).astype(int)
            m_default = compute_per_label_metrics(
                test_true, test_binary_default, test_probs)
            print(f'  [Default thr=0.5]  '
                  f'Macro F1={m_default["macro_f1"]:.4f}  '
                  f'Macro AUC={m_default.get("macro_auc", "N/A")}')
            summary_rows.append(
                {'model': model_name, 'dataset': dataset_name, **m_default})

            # ── Threshold calibration on val ──────────────────────────────────
            print('  Calibrating thresholds on validation set...')
            best_thrs = find_optimal_thresholds(val_true, val_probs)

            # ── Apply calibrated thresholds on test ───────────────────────────
            test_binary_calib = apply_thresholds(test_probs, best_thrs)
            m_calib = compute_per_label_metrics(
                test_true, test_binary_calib, test_probs)
            print(f'  [Calibrated]       '
                  f'Macro F1={m_calib["macro_f1"]:.4f}  '
                  f'Macro AUC={m_calib.get("macro_auc", "N/A")}')
            calib_rows.append(
                {'model': model_name, 'dataset': dataset_name,
                 'best_thresholds': best_thrs, **m_calib})

            # ── Plots ─────────────────────────────────────────────────────────
            plot_threshold_f1_curves(
                val_true, val_probs, model_name, dataset_name, OUTPUT_DIR)
            plot_threshold_improvement(
                m_default, m_calib, model_name, dataset_name, OUTPUT_DIR)

            del model_obj
            torch.cuda.empty_cache()
            gc.collect()

    # ── Summary CSVs ─────────────────────────────────────────────────────────
    print('\n── SAVING SUMMARY ──────────────────────────────────────────────────')

    default_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'best_thresholds'}
        for r in summary_rows
    ])
    calib_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'best_thresholds'}
        for r in calib_rows
    ])
    thr_df = pd.DataFrame([
        {'model': r['model'], 'dataset': r['dataset'], **r['best_thresholds']}
        for r in calib_rows
    ])

    default_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_default.csv'),
                      index=False)
    calib_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_calibrated.csv'),
                    index=False)
    thr_df.to_csv(os.path.join(OUTPUT_DIR, 'optimal_thresholds.csv'),
                  index=False)

    print('\nOptimal thresholds per model and dataset:')
    print(thr_df.to_string(index=False))

    print('\n' + '=' * 70)
    print(f'End  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 70)
