import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
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

PROTBERT_PTH = '../results/protbert_multilabel_dramp_final.pth'
ESM2_PTH = '../results/esm2_multilabel_dramp_final.pth'
PROTT5_PTH = '../results/prott5_multilabel_dramp_final.pth'

DRAMP_TEST = '../data/dramp_test.csv'
DBAMP_TEST = '../data/dbamp_test.csv'

USE_SAVED_WEIGHTS = True
OUTPUT_DIR = '../results/label_error_profile_dramp'
BATCH_SIZE = 32
THRESHOLD = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_COLS = ['antimicrobial', 'antiviral', 'antifungal', 'anticancer']

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

AROMATICITY = {'F', 'W', 'Y', 'H'}


def seq_hydrophobicity(seq: str) -> float:
    vals = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq.upper()]
    return float(np.mean(vals)) if vals else 0.0


def seq_charge(seq: str) -> int:
    return sum(CHARGE.get(aa, 0) for aa in seq.upper())


def seq_length(seq: str) -> int:
    return len(seq)


def seq_aromaticity(seq: str) -> float:
    """Fraction of aromatic residues (F, W, Y, H)."""
    if not seq:
        return 0.0
    return sum(1 for aa in seq.upper() if aa in AROMATICITY) / len(seq)


class ProtBERTClassifier(nn.Module):
    def __init__(self, n_classes: int = len(LABEL_COLS), dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class ESM2Classifier(nn.Module):
    def __init__(self, model_name: str = 'facebook/esm2_t6_8M_UR50D',
                 n_classes: int = len(LABEL_COLS), dropout: float = 0.3):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.esm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class ProtT5Classifier(nn.Module):
    def __init__(self, model_name: str = 'Rostlab/prot_t5_xl_half_uniref50-enc',
                 n_classes: int = len(LABEL_COLS), dropout: float = 0.3,
                 freeze_t5: bool = False):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.t5.config.d_model, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        mean_emb = torch.sum(embeddings * mask_exp, 1) / torch.clamp(
            mask_exp.sum(1), min=1e-9)
        return self.classifier(self.dropout(mean_emb))


def load_model(model_obj: nn.Module, pth_path: str, label: str) -> nn.Module:
    if USE_SAVED_WEIGHTS and os.path.exists(pth_path):
        model_obj.load_state_dict(torch.load(pth_path, map_location=device))
        print(f'Weights loaded: {pth_path}')
    else:
        print(f'WARNING: {pth_path} not found – using pretrained weights only')
    model_obj.eval().to(device)
    print(f'{label} ready.')
    return model_obj


def format_sequence(sequence: str, model_type: str) -> str:
    if model_type == 'bert':
        return ' '.join(list(sequence))
    elif model_type == 't5':
        seq = sequence.replace('U', 'X').replace('Z', 'X') \
            .replace('O', 'X').replace('B', 'X')
        return f"<AA2fold> {' '.join(list(seq))}"
    return sequence  # ESM-2


def run_inference(df: pd.DataFrame, model: nn.Module,
                  tokenizer, model_type: str):
    """
    Returns
    -------
    preds : (N, n_labels) binary array after sigmoid + threshold
    probs : (N, n_labels) sigmoid probabilities
    """
    all_probs = []
    sequences = df['sequence'].tolist()

    for start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[start: start + BATCH_SIZE]
        formatted = [format_sequence(s, model_type) for s in batch]

        encoding = tokenizer(
            formatted,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())

        done = min(start + BATCH_SIZE, len(sequences))
        if (start // BATCH_SIZE + 1) % 5 == 0:
            print(f'{done}/{len(sequences)} processed...')

    probs_arr = np.array(all_probs)  # (N, n_labels)
    preds_arr = (probs_arr >= THRESHOLD).astype(int)
    return preds_arr, probs_arr


def build_label_error_df(df: pd.DataFrame,
                         true_labels: np.ndarray,
                         pred_labels: np.ndarray,
                         probs: np.ndarray,
                         model_name: str,
                         dataset_name: str) -> pd.DataFrame:
    """
    One row per (sequence x label) combination.
    error_type: TP | TN | FP | FN  (per label)
    """
    records = []
    for i, seq in enumerate(df['sequence']):
        phys = {
            'length': seq_length(seq),
            'charge': seq_charge(seq),
            'hydrophobicity': seq_hydrophobicity(seq),
            'aromaticity': seq_aromaticity(seq),
        }
        for j, label in enumerate(LABEL_COLS):
            t = int(true_labels[i, j])
            p = int(pred_labels[i, j])

            if t == 1 and p == 1:
                error_type = 'TP'
            elif t == 0 and p == 0:
                error_type = 'TN'
            elif t == 0 and p == 1:
                error_type = 'FP'
            else:
                error_type = 'FN'

            records.append({
                'model': model_name,
                'dataset': dataset_name,
                'label': label,
                'sequence': seq,
                'true': t,
                'pred': p,
                'prob': round(float(probs[i, j]), 4),
                'error_type': error_type,
                'is_error': error_type in ('FP', 'FN'),
                **phys,
            })

    return pd.DataFrame(records)


PROPERTIES = [
    ('length', 'Sequence length (AA)'),
    ('hydrophobicity', 'Avg hydrophobicity (KD)'),
    ('charge', 'Total charge'),
    ('aromaticity', 'Aromaticity (fraction)'),
]

ERROR_COLORS = {
    'TP': '#2ecc71',
    'TN': '#3498db',
    'FP': '#e67e22',
    'FN': '#e74c3c',
}

LABEL_COLORS = {
    'antimicrobial': '#3498db',
    'antiviral': '#e74c3c',
    'antifungal': '#2ecc71',
    'anticancer': '#9b59b6',
}


def safe_name(s: str) -> str:
    return s.lower().replace('-', '_').replace(' ', '_')


def plot_per_label_error_profile(error_df, model_name, dataset_name, output_dir):
    """
    4 rows (one per label) x 4 columns (one per property).
    Each cell: overlapping histograms for TP / TN / FP / FN.
    """
    fig, axes = plt.subplots(
        len(LABEL_COLS), len(PROPERTIES),
        figsize=(20, 16), sharex='col',
    )
    fig.suptitle(
        f'{model_name}  |  {dataset_name}\n'
        f'Per-label physicochemical error profile',
        fontsize=14, fontweight='bold', y=1.01,
    )

    for row, label in enumerate(LABEL_COLS):
        sub_label = error_df[error_df['label'] == label]

        for col, (prop, xlabel) in enumerate(PROPERTIES):
            ax = axes[row, col]

            for etype, color in ERROR_COLORS.items():
                vals = sub_label[sub_label['error_type'] == etype][prop]
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=20, alpha=0.50, color=color,
                        label=f'{etype} (n={len(vals)})', density=True)
                ax.axvline(vals.mean(), color=color,
                           linewidth=1.8, linestyle='--', alpha=0.9)

            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold',
                              color=LABEL_COLORS[label])
            if row == 0:
                ax.set_title(xlabel, fontsize=10, fontweight='bold')
            if row == len(LABEL_COLS) - 1:
                ax.set_xlabel(xlabel, fontsize=9)

            ax.grid(alpha=0.3)
            if row == 0 and col == len(PROPERTIES) - 1:
                ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f'label_error_profile_{safe_name(model_name)}_{safe_name(dataset_name)}.png',
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_fn_physicochemical_per_label(error_df, model_name, dataset_name, output_dir):
    """
    For each label: compare physicochemical distribution of FN vs TP.
    Mann-Whitney U test for significance.
    """
    fig, axes = plt.subplots(
        len(LABEL_COLS), len(PROPERTIES),
        figsize=(20, 14),
    )
    fig.suptitle(
        f'{model_name}  |  {dataset_name}\n'
        f'False Negatives vs True Positives per label',
        fontsize=13, fontweight='bold', y=1.01,
    )

    for row, label in enumerate(LABEL_COLS):
        sub = error_df[error_df['label'] == label]
        tp_sub = sub[sub['error_type'] == 'TP']
        fn_sub = sub[sub['error_type'] == 'FN']

        for col, (prop, xlabel) in enumerate(PROPERTIES):
            ax = axes[row, col]
            tp_vals = tp_sub[prop].values
            fn_vals = fn_sub[prop].values

            for vals, color, etag in [
                (tp_vals, '#2ecc71', f'TP (n={len(tp_vals)})'),
                (fn_vals, '#e74c3c', f'FN (n={len(fn_vals)})'),
            ]:
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=15, alpha=0.60, color=color,
                        label=etag, density=True)
                ax.axvline(vals.mean(), color=color,
                           linewidth=2, linestyle='--')

            if len(tp_vals) > 1 and len(fn_vals) > 1:
                _, p_val = mannwhitneyu(tp_vals, fn_vals, alternative='two-sided')
                sig = ('***' if p_val < 0.001 else
                       '**' if p_val < 0.01 else
                       '*' if p_val < 0.05 else 'ns')
                ax.set_title(f'{sig}  p={p_val:.3f}', fontsize=8,
                             color='#c0392b' if sig != 'ns' else 'gray')

            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold',
                              color=LABEL_COLORS[label])
            if row == len(LABEL_COLS) - 1:
                ax.set_xlabel(xlabel, fontsize=9)

            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f'fn_vs_tp_{safe_name(model_name)}_{safe_name(dataset_name)}.png',
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_confidence_per_label(error_df, model_name, dataset_name, output_dir):
    """Scatter: sigmoid confidence vs length, coloured by error type, per label."""
    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(20, 5))
    fig.suptitle(
        f'{model_name}  |  {dataset_name}\n'
        f'Model confidence vs sequence length (per label)',
        fontsize=13, fontweight='bold',
    )

    for ax, label in zip(axes, LABEL_COLS):
        sub = error_df[error_df['label'] == label]
        for etype, color in ERROR_COLORS.items():
            s = sub[sub['error_type'] == etype]
            if len(s) == 0:
                continue
            ax.scatter(s['length'], s['prob'],
                       c=color, alpha=0.45, s=15,
                       label=f'{etype} (n={len(s)})')

        ax.axhline(THRESHOLD, color='gray', linestyle='--',
                   linewidth=1, alpha=0.7, label=f'Threshold ({THRESHOLD})')
        ax.set_title(label, fontsize=11, fontweight='bold',
                     color=LABEL_COLORS[label])
        ax.set_xlabel('Length (AA)', fontsize=9)
        ax.set_ylabel('Sigmoid probability', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f'confidence_{safe_name(model_name)}_{safe_name(dataset_name)}.png',
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_fn_rate_heatmap(summary_rows, output_dir):
    """Heatmap: FN rate per model x label, one subplot per dataset."""
    df = pd.DataFrame(summary_rows)
    datasets = df['dataset'].unique()

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(8 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle('False Negative rate per model x label (%)',
                 fontsize=13, fontweight='bold')

    for ax, dataset in zip(axes, datasets):
        pivot = (
                df[df['dataset'] == dataset]
                .pivot(index='model', columns='label', values='fn_rate') * 100
        )
        pivot = pivot[[l for l in LABEL_COLS if l in pivot.columns]]
        sns.heatmap(
            pivot, ax=ax,
            annot=True, fmt='.1f', cmap='Reds',
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'FN rate (%)'},
            vmin=0,
        )
        ax.set_title(dataset, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=20)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fn_rate_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_three_model_fn_comparison(all_error_dfs, dataset_name, output_dir):
    """
    For each label: FN hydrophobicity distribution across all three models.
    Reveals whether different models fail on the same physicochemical subset.
    """
    model_names = ['ESM-2', 'ProtBERT', 'ProtT5']
    model_colors = {'ESM-2': '#3498db', 'ProtBERT': '#e74c3c', 'ProtT5': '#2ecc71'}
    prop, p_label = 'hydrophobicity', 'Avg hydrophobicity (KD)'

    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(20, 5))
    fig.suptitle(
        f'False Negatives – hydrophobicity comparison across models\n{dataset_name}',
        fontsize=13, fontweight='bold',
    )

    for ax, label in zip(axes, LABEL_COLS):
        for mname in model_names:
            key = (mname, dataset_name)
            if key not in all_error_dfs:
                continue
            fn_vals = all_error_dfs[key]
            fn_vals = fn_vals[
                (fn_vals['label'] == label) &
                (fn_vals['error_type'] == 'FN')
                ][prop]

            if len(fn_vals) == 0:
                continue
            ax.hist(fn_vals, bins=15, alpha=0.55,
                    color=model_colors[mname],
                    label=f'{mname} (n={len(fn_vals)})', density=True)
            ax.axvline(fn_vals.mean(), color=model_colors[mname],
                       linewidth=2, linestyle='--', alpha=0.9)

        ax.set_title(label, fontsize=11, fontweight='bold',
                     color=LABEL_COLORS[label])
        ax.set_xlabel(p_label, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f'fn_3models_{safe_name(dataset_name)}.png',
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def compute_label_summary(error_df, model_name, dataset_name):
    rows = []
    for label in LABEL_COLS:
        sub = error_df[error_df['label'] == label]
        tp = (sub['error_type'] == 'TP').sum()
        tn = (sub['error_type'] == 'TN').sum()
        fp = (sub['error_type'] == 'FP').sum()
        fn = (sub['error_type'] == 'FN').sum()

        fn_rate = fn / (fn + tp + 1e-9)
        fp_rate = fp / (fp + tn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        rows.append({
            'model': model_name,
            'dataset': dataset_name,
            'label': label,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'fn_rate': round(fn_rate, 4),
            'fp_rate': round(fp_rate, 4),
        })
    return rows


if __name__ == '__main__':
    print('LABEL ERROR PROFILE  –  ProtBERT · ESM-2 · ProtT5')
    print(f'Start : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
        print(f'VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = {}
    for name, path in [('DRAMP', DRAMP_TEST), ('dbAMP', DBAMP_TEST)]:
        if not os.path.exists(path):
            print(f'WARNING: {path} not found – skipping {name}')
            continue
        df = pd.read_csv(path)
        print(f'\n{name}: {len(df):,} sequences')
        for lc in LABEL_COLS:
            print(f'  {lc:<18}: {df[lc].sum():>6,} positives')
        datasets[name] = df

    if not datasets:
        raise RuntimeError('No test datasets found. Check paths.')

    all_error_dfs = {}  # {(model_name, dataset_name): error_df}
    summary_rows = []
    all_error_rows = []

    transformer_configs = [
        (
            'ESM-2',
            lambda: ESM2Classifier(),
            lambda: AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'),
            ESM2_PTH, 'esm',
        ),
        (
            'ProtBERT',
            lambda: ProtBERTClassifier(),
            lambda: BertTokenizer.from_pretrained(
                'Rostlab/prot_bert', do_lower_case=False),
            PROTBERT_PTH, 'bert',
        ),
        (
            'ProtT5',
            lambda: ProtT5Classifier(freeze_t5=True),
            lambda: T5Tokenizer.from_pretrained(
                'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False),
            PROTT5_PTH, 't5',
        ),
    ]

    for model_name, model_fn, tok_fn, pth, mtype in transformer_configs:
        print(f'{model_name}')

        tokenizer = tok_fn()
        model = load_model(model_fn(), pth, model_name)

        for dataset_name, df in datasets.items():
            print(f'\n [{dataset_name}]')
            true_labels = df[LABEL_COLS].values.astype(int)  # (N, 4)

            preds, probs = run_inference(df, model, tokenizer, mtype)

            error_df = build_label_error_df(
                df, true_labels, preds, probs, model_name, dataset_name)
            all_error_dfs[(model_name, dataset_name)] = error_df
            all_error_rows.append(error_df)

            label_rows = compute_label_summary(error_df, model_name, dataset_name)
            summary_rows.extend(label_rows)

            print(f'\n  {"Label":<18} {"P":>6} {"R":>6} {"F1":>6} '
                  f'{"FN":>5} {"FP":>5} {"FN%":>7}')
            print(f'  {"-" * 60}')
            for r in label_rows:
                print(f'  {r["label"]:<18} {r["precision"]:>6.3f} '
                      f'{r["recall"]:>6.3f} {r["f1"]:>6.3f} '
                      f'{r["FN"]:>5} {r["FP"]:>5} '
                      f'{r["fn_rate"] * 100:>6.1f}%')

            plot_per_label_error_profile(
                error_df, model_name, dataset_name, OUTPUT_DIR)
            plot_fn_physicochemical_per_label(
                error_df, model_name, dataset_name, OUTPUT_DIR)
            plot_confidence_per_label(
                error_df, model_name, dataset_name, OUTPUT_DIR)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print('Cross-model comparison')

    for dataset_name in datasets:
        plot_three_model_fn_comparison(all_error_dfs, dataset_name, OUTPUT_DIR)

    plot_fn_rate_heatmap(summary_rows, OUTPUT_DIR)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'label_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'\nSummary table saved: {summary_path}')

    full_errors = pd.concat(all_error_rows, ignore_index=True)
    errors_only = full_errors[full_errors['is_error']]
    errors_path = os.path.join(OUTPUT_DIR, 'misclassified_sequences.csv')
    errors_only.to_csv(errors_path, index=False)
    print(f'Misclassified sequences saved: {errors_path} '
          f'({len(errors_only):,} rows)')

    print('SUMMARY TABLE')
    print(summary_df.to_string(index=False))

    print('\nDONE!')
    print(f'End   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
