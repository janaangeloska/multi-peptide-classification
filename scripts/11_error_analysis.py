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

PROTBERT_DBAMP_PTH = '../results/protbert_multilabel_dbamp_final.pth'
PROTBERT_DRAMP_PTH = '../results/protbert_multilabel_dramp_final.pth'
ESM2_DBAMP_PTH = '../results/esm2_multilabel_dbamp_final.pth'
ESM2_DRAMP_PTH = '../results/esm2_multilabel_dramp_final.pth'
PROTT5_DBAMP_PTH = '../results/prott5_multilabel_dbamp_final.pth'
PROTT5_DRAMP_PTH = '../results/prott5_multilabel_dramp_final.pth'

DBAMP_TEST_CSV = '../data/dbamp_test.csv'
DRAMP_TEST_CSV = '../data/dramp_test.csv'

USE_SAVED_WEIGHTS = True
OUTPUT_DIR = '../results/error_analysis'
BATCH_SIZE = 64
THRESHOLD = 0.5

# Label definition — must match training scripts
LABEL_COLS = ['antimicrobial', 'antiviral', 'antifungal', 'anticancer']
N_LABELS = len(LABEL_COLS)

LABEL_COLORS = {
    'antimicrobial': '#e74c3c',
    'antiviral': '#3498db',
    'antifungal': '#2ecc71',
    'anticancer': '#9b59b6',
}

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
    print(f'  {label} подготвен.')
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
# АНАЛИЗА НА ГРЕШКИ
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
                 fontsize=12, fontweight='bold')

    mcm = multilabel_confusion_matrix(true_labels, pred_binary)
    for ax, (i, lbl) in zip(axes, enumerate(label_cols)):
        cm = mcm[i]
        tn, fp, fn, tp = cm.ravel()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'],
                    linewidths=0.5)
        fn_r = fn / (fn + tp + 1e-9)
        fp_r = fp / (fp + tn + 1e-9)
        ax.set_title(f'{lbl}\nFN rate={fn_r:.2%}  FP rate={fp_r:.2%}',
                     fontsize=9, fontweight='bold',
                     color=LABEL_COLORS.get(lbl, 'black'))
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)

        ax.text(0.5, 1.5, 'FN', ha='center', va='center', fontsize=9,
                color='#e74c3c', fontweight='bold')
        ax.text(1.5, 0.5, 'FP', ha='center', va='center', fontsize=9,
                color='#e67e22', fontweight='bold')

    plt.tight_layout()
    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'cm_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_error_profiles(error_df, model_name, dataset_name, output_dir,
                        label_cols=LABEL_COLS):
    colors = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e67e22', 'FN': '#e74c3c'}
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
            fontsize=12, fontweight='bold'
        )
        for ax, (col, xlabel) in zip(axes, props):
            for etype, color in colors.items():
                subset = error_df[error_df[f'etype_{lbl}'] == etype][col]
                if len(subset) == 0:
                    continue
                ax.hist(subset, bins=20, alpha=0.55, color=color,
                        label=f'{etype} (n={len(subset)})', density=True)
                ax.axvline(subset.mean(), color=color, linewidth=2,
                           linestyle='--', alpha=0.8)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel('Густина', fontsize=10)
            ax.set_title(xlabel, fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}_{lbl}'
        path = os.path.join(output_dir, f'error_profiles_{tag}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_confidence_errors(error_df, model_name, dataset_name, output_dir,
                           label_cols=LABEL_COLS):
    colors = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e67e22', 'FN': '#e74c3c'}
    n = len(label_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle(f'{model_name} — {dataset_name}\nConfidence vs length',
                 fontsize=12, fontweight='bold')

    for ax, lbl in zip(axes, label_cols):
        for etype, color in colors.items():
            sub = error_df[error_df[f'etype_{lbl}'] == etype]
            if len(sub) == 0:
                continue
            ax.scatter(sub['length'], sub[f'prob_{lbl}'],
                       c=color, alpha=0.5, s=18,
                       label=f'{etype} (n={len(sub)})')

        ax.axhline(THRESHOLD, color='gray', linestyle='--', linewidth=1,
                   alpha=0.7, label=f'Threshold ({THRESHOLD})')
        ax.set_xlabel('Length (AA)', fontsize=10)
        ax.set_ylabel(f'P({lbl})', fontsize=10)
        ax.set_title(lbl, fontsize=10, fontweight='bold',
                     color=LABEL_COLORS.get(lbl, 'black'))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

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
        sns.heatmap(fn_matrix, annot=True, fmt='.2%', cmap='Reds',
                    xticklabels=label_cols, yticklabels=models,
                    linewidths=0.5, ax=ax, vmin=0, vmax=1)
        ax.set_title(f'FN Rate per model and label [{ds_name}]',
                     fontsize=12, fontweight='bold')
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
        model_colors = {'ESM-2': '#3498db', 'ProtBERT': '#e74c3c', 'ProtT5': '#2ecc71'}

        fig, ax = plt.subplots(figsize=(12, 5))
        for offset, mname in zip(
                np.linspace(-width, width, len(models)), models):
            if mname not in subset:
                continue
            vals = [subset[mname].get(f'{lbl}_f1', 0.0) or 0.0
                    for lbl in label_cols]
            bars = ax.bar(x + offset, vals, width,
                          label=mname,
                          color=model_colors.get(mname, 'gray'),
                          alpha=0.8, edgecolor='white')
            ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

        ax.set_title(f'Per-label F1 per model [{ds_name}]',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(label_cols, fontsize=10)
        ax.set_ylabel('F1 Score')
        ax.set_ylim([0, 1.12])
        ax.legend(title='Model', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f'per_label_f1_{ds_name.lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_n_label_errors_distribution(error_df, model_name,
                                     dataset_name, output_dir):
    counts = error_df['n_label_errors'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color='#8e44ad', alpha=0.8, edgecolor='white')
    ax.bar_label(bars, fontsize=9, padding=2)
    ax.set_xlabel('Number of wrong labels per sequence', fontsize=11)
    ax.set_ylabel('Number of sequences', fontsize=11)
    ax.set_title(f'{model_name} — {dataset_name}\n'
                 f'Distribution of errors per sequence',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(counts.index)
    ax.grid(axis='y', alpha=0.3)
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
        ('subset_accuracy', 'Subset Accuracy', '#2ecc71'),
        ('macro_f1', 'Macro F1', '#3498db'),
        ('macro_auc', 'Macro AUC', '#9b59b6'),
        ('hamming_loss', 'Hamming Loss', '#e74c3c'),
    ]

    x = np.arange(len(models))
    width = 0.35
    ds_colors = {'dbAMP': '#3498db', 'DRAMP': '#e74c3c'}

    fig, axes = plt.subplots(1, len(metrics_cfg),
                             figsize=(5 * len(metrics_cfg), 5))
    fig.suptitle('Aggregate Metrics per model and dataset',
                 fontsize=13, fontweight='bold')

    for ax, (metric, title, _) in zip(axes, metrics_cfg):
        for offset, ds in zip([-width / 2, width / 2], datasets):
            vals = []
            for m in models:
                row = df[(df['model'] == m) & (df['dataset'] == ds)]
                vals.append(float(row[metric].values[0])
                            if len(row) and row[metric].values[0] is not None
                            else 0.0)
            bars = ax.bar(x + offset, vals, width, label=ds,
                          color=ds_colors.get(ds, 'gray'),
                          alpha=0.8, edgecolor='white')
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10, ha='right')
        ax.set_ylim([0, 1.12])
        ax.legend(title='Dataset', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'aggregate_metrics_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Зачувано: {path}')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print('=' * 70)
    print('ERROR ANALYSIS — ProtBERT · ESM-2 · ProtT5  [MULTILABEL]')
    print(f'Labels: {LABEL_COLS}')
    print(f'Start : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
        print(f'VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('=' * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = {}
    for name, csv_path in [('dbAMP', DBAMP_TEST_CSV), ('DRAMP', DRAMP_TEST_CSV)]:
        df = pd.read_csv(csv_path)
        print(f'\n{name}: {len(df)} секвенци')
        for lbl in LABEL_COLS:
            print(f'  {lbl}: {df[lbl].sum()} positive')
        datasets[name] = df

    pth_map = {
        'ESM-2': {'dbAMP': ESM2_DBAMP_PTH, 'DRAMP': ESM2_DRAMP_PTH},
        'ProtBERT': {'dbAMP': PROTBERT_DBAMP_PTH, 'DRAMP': PROTBERT_DRAMP_PTH},
        'ProtT5': {'dbAMP': PROTT5_DBAMP_PTH, 'DRAMP': PROTT5_DRAMP_PTH},
    }

    model_configs = [
        ('ESM-2',
         lambda: ESM2MultilabelClassifier(),
         lambda: AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'),
         'esm'),
        ('ProtBERT',
         lambda: ProtBERTMultilabelClassifier(),
         lambda: BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False),
         'bert'),
        ('ProtT5',
         lambda: ProtT5MultilabelClassifier(freeze_t5=True),
         lambda: T5Tokenizer.from_pretrained(
             'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False),
         't5'),
    ]

    all_error_dfs = []
    summary_rows = []

    for model_name, model_fn, tok_fn, mtype in model_configs:
        print(f'\n{"─" * 60}')
        print(f'── {model_name}')
        print(f'{"─" * 60}')

        tokenizer = tok_fn()

        for dataset_name, df in datasets.items():
            print(f'\n  [{dataset_name}]')
            pth = pth_map[model_name][dataset_name]
            model_obj = load_model(model_fn(), pth, f'{model_name} [{dataset_name}]')

            true_labels = df[LABEL_COLS].values.astype(int)
            pred_probs, pred_binary = run_inference(df, model_obj, tokenizer, mtype)

            m = compute_per_label_metrics(true_labels, pred_binary, pred_probs)
            print(f'  Subset Acc={m["subset_accuracy"]:.3f}  '
                  f'Macro F1={m["macro_f1"]:.3f}  '
                  f'Macro AUC={m.get("macro_auc", "N/A")}  '
                  f'Hamming={m["hamming_loss"]:.3f}')
            for lbl in LABEL_COLS:
                print(f'    {lbl:15s}: '
                      f'F1={m[f"{lbl}_f1"]:.3f}  '
                      f'AUC={m.get(f"{lbl}_auc", "N/A")}  '
                      f'FN rate={m[f"{lbl}_fn_rate"]:.3f}  '
                      f'FP rate={m[f"{lbl}_fp_rate"]:.3f}  '
                      f'TP={m[f"{lbl}_tp"]}  FN={m[f"{lbl}_fn"]}')

            print(f'\n  Classification Report [{dataset_name}]:')
            print(classification_report(true_labels, pred_binary,
                                        target_names=LABEL_COLS, zero_division=0))

            plot_multilabel_confusion_matrices(
                true_labels, pred_binary, model_name, dataset_name, OUTPUT_DIR)

            error_df = build_error_df(
                df, true_labels, pred_binary, pred_probs, model_name, dataset_name)
            all_error_dfs.append(error_df)

            plot_error_profiles(error_df, model_name, dataset_name, OUTPUT_DIR)
            plot_confidence_errors(error_df, model_name, dataset_name, OUTPUT_DIR)
            plot_n_label_errors_distribution(error_df, model_name, dataset_name, OUTPUT_DIR)

            row = {'model': model_name, 'dataset': dataset_name, **m}
            summary_rows.append(row)

            del model_obj
            torch.cuda.empty_cache()
            gc.collect()

    print('\n── COMPARISON - all three models ──────────────────────────────────────')
    plot_fn_rate_comparison(summary_rows, OUTPUT_DIR)
    plot_per_label_f1_comparison(summary_rows, OUTPUT_DIR)
    plot_aggregate_metrics_summary(summary_rows, OUTPUT_DIR)

    full_error_df = pd.concat(all_error_dfs, ignore_index=True)
    errors_only = full_error_df[full_error_df['any_error']]
    csv_path = os.path.join(OUTPUT_DIR, 'error_analysis_summary.csv')
    errors_only.to_csv(csv_path, index=False)
    print(f'\nTotal sequences with ≥1 wrong labels: {len(errors_only)}')
    print(f'Saved: {csv_path}')

    print('\n── SUMMARY TABLE ──────────────────────────────────────────────────')
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df[['model', 'dataset', 'subset_accuracy', 'macro_f1',
                      'macro_auc', 'hamming_loss']].to_string(index=False))
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)

    print('\n' + '=' * 70)
    print(f'End  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 70)
