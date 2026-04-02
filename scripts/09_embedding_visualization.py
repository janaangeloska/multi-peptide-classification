import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
from transformers import (
    BertTokenizer, BertModel,
    AutoTokenizer, EsmModel,
    T5Tokenizer, T5EncoderModel
)
import warnings
import os
import gc
from datetime import datetime
from itertools import combinations

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
OUTPUT_DIR = '../results/embedding_viz'

LABEL_COLS = ['antimicrobial', 'antiviral', 'antifungal', 'anticancer']
N_LABELS = len(LABEL_COLS)

MAX_SAMPLES = None

TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RANDOM_STATE = 42

LABEL_COLORS = {
    'antimicrobial': '#e74c3c',
    'antiviral': '#3498db',
    'antifungal': '#2ecc71',
    'anticancer': '#9b59b6',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProtBERTMultilabelClassifier(nn.Module):
    def __init__(self, n_labels=N_LABELS, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(self.dropout(cls))
        return logits, cls  # also return embedding


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
        logits = self.classifier(self.dropout(cls))
        return logits, cls


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
        logits = self.classifier(self.dropout(mean_emb))
        return logits, mean_emb


def load_model(model_obj, pth_path, label):
    if USE_SAVED_WEIGHTS and os.path.exists(pth_path):
        model_obj.load_state_dict(torch.load(pth_path, map_location=device))
        print(f'  Weights: {pth_path}')
    else:
        print(f'  Warning: {pth_path} does not exist - overtrained weights')
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


def label_combo_name(row, label_cols=LABEL_COLS):
    abbrev = {'antimicrobial': 'AM', 'antiviral': 'AV',
              'antifungal': 'AF', 'anticancer': 'AC'}
    active = [abbrev[c] for c in label_cols if row[c] == 1]
    return '+'.join(active) if active else 'none'


def extract_embeddings(df, model, tokenizer, model_type, batch_size=32):
    all_embeddings = []
    all_probs = []

    sequences = df['sequence'].tolist()
    n = len(sequences)
    print(f'  Extracting {n} embeddings (batch={batch_size})...')

    for start in range(0, n, batch_size):
        batch_seqs = sequences[start: start + batch_size]
        formatted = [format_sequence(s, model_type) for s in batch_seqs]

        encoding = tokenizer(
            formatted,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits, emb = model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.sigmoid(logits)

        all_embeddings.append(emb.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        if (start // batch_size + 1) % 10 == 0:
            done = min(start + batch_size, n)
            print(f'    {done}/{n} sequences done...')

    embeddings = np.vstack(all_embeddings)
    pred_probs = np.vstack(all_probs)
    pred_binary = (pred_probs >= 0.5).astype(int)
    print(f'  Embedding shape: {embeddings.shape}')
    return embeddings, pred_probs, pred_binary


def reduce_embeddings(embeddings):
    print('  PCA (50 components)...')
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    n_components = min(50, emb_scaled.shape[0], emb_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(emb_scaled)
    var_exp = pca.explained_variance_ratio_[:10].sum() * 100
    print(f'  PCA - top 10 components explain {var_exp:.1f}% variance')

    print(f'  t-SNE (perplexity={TSNE_PERPLEXITY})...')
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                n_iter=TSNE_ITERATIONS, random_state=RANDOM_STATE, verbose=0)
    emb_tsne = tsne.fit_transform(emb_pca)

    print(f'  UMAP (n_neighbors={UMAP_NEIGHBORS})...')
    reducer = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        n_components=2, random_state=RANDOM_STATE)
    emb_umap = reducer.fit_transform(emb_pca)

    return emb_tsne, emb_umap, emb_pca, pca


def compute_multilabel_separation_metrics(emb_2d, true_labels, pred_binary,
                                          label_cols=LABEL_COLS):
    metrics = {}

    for i, lbl in enumerate(label_cols):
        col = true_labels[:, i]
        if len(np.unique(col)) > 1:
            metrics[f'sil_true_{lbl}'] = round(silhouette_score(emb_2d, col), 4)
        else:
            metrics[f'sil_true_{lbl}'] = None

    sil_vals = [v for v in [metrics[f'sil_true_{l}'] for l in label_cols] if v is not None]
    metrics['mean_sil_true'] = round(np.mean(sil_vals), 4) if sil_vals else None

    exact_match = np.all(true_labels == pred_binary, axis=1).mean() * 100
    metrics['subset_accuracy_pct'] = round(exact_match, 2)

    for i, lbl in enumerate(label_cols):
        acc = (true_labels[:, i] == pred_binary[:, i]).mean() * 100
        metrics[f'acc_{lbl}'] = round(acc, 2)

    return metrics


def _scatter_by_label(ax, coords, label_vec, label_name, color, title):
    pos_mask = label_vec == 1
    neg_mask = ~pos_mask

    ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
               c='lightgray', alpha=0.35, s=14, linewidths=0, label=f'no {label_name}')
    ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1],
               c=color, alpha=0.65, s=20, linewidths=0, label=f'{label_name} (+)')

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.2)


def _scatter_by_combo(ax, coords, true_labels_df, title):
    combos = true_labels_df.apply(label_combo_name, axis=1).values
    unique_combos = sorted(set(combos))
    palette = plt.cm.tab20.colors
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique_combos)}

    for combo in unique_combos:
        mask = combos == combo
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap[combo]], alpha=0.6, s=16,
                   linewidths=0, label=f'{combo} (n={mask.sum()})')

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='lower right', ncol=2)
    ax.grid(alpha=0.2)


def plot_single_model(model_name, dataset_name, emb_tsne, emb_umap,
                      true_labels_arr, pred_binary, metrics_tsne, metrics_umap,
                      output_dir, label_cols=LABEL_COLS):
    true_df = pd.DataFrame(true_labels_arr, columns=label_cols)

    n_cols = len(label_cols) + 1  # 4 labels + 1 combo
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    fig.suptitle(
        f'{model_name} [{dataset_name}] — Multilabel Embedding Projection\n'
        f't-SNE mean sil: {metrics_tsne.get("mean_sil_true", "N/A")} | '
        f'UMAP mean sil: {metrics_umap.get("mean_sil_true", "N/A")} | '
        f'Subset Acc: {metrics_tsne.get("subset_accuracy_pct", "N/A")}%',
        fontsize=12, fontweight='bold', y=1.01
    )

    for col_i, lbl in enumerate(label_cols):
        color = LABEL_COLORS[lbl]
        sil_t = metrics_tsne.get(f'sil_true_{lbl}', 'N/A')
        sil_u = metrics_umap.get(f'sil_true_{lbl}', 'N/A')

        _scatter_by_label(axes[0, col_i], emb_tsne, true_labels_arr[:, col_i],
                          lbl, color, f't-SNE  {lbl}\nsil={sil_t}')
        _scatter_by_label(axes[1, col_i], emb_umap, true_labels_arr[:, col_i],
                          lbl, color, f'UMAP  {lbl}\nsil={sil_u}')

    _scatter_by_combo(axes[0, -1], emb_tsne, true_df, 't-SNE — label combos')
    _scatter_by_combo(axes[1, -1], emb_umap, true_df, 'UMAP — label combos')

    plt.tight_layout()
    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'embedding_{tag}_per_label.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_three_model_comparison(model_results, dataset_name, output_dir,
                                label_cols=LABEL_COLS):
    models = ['ESM-2', 'ProtBERT', 'ProtT5']
    n_rows = len(label_cols)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    fig.suptitle(
        f'ESM-2 vs ProtBERT vs ProtT5 — t-SNE per-label [{dataset_name}]',
        fontsize=14, fontweight='bold', y=1.01
    )

    for row_i, lbl in enumerate(label_cols):
        for col_i, mname in enumerate(models):
            key = (mname, dataset_name)
            if key not in model_results:
                axes[row_i, col_i].axis('off')
                continue
            res = model_results[key]
            sil = res['metrics_tsne'].get(f'sil_true_{lbl}', 'N/A')
            color = LABEL_COLORS[lbl]
            _scatter_by_label(
                axes[row_i, col_i],
                res['emb_tsne'],
                res['true_labels'][:, label_cols.index(lbl)],
                lbl, color,
                f'{mname} | {lbl}\nsil={sil}'
            )

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'embedding_comparison_3models_{dataset_name.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_pca_variance(model_name, dataset_name, pca, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-',
            color='steelblue', linewidth=2, markersize=5)
    ax.axhline(90, color='red', linestyle='--', alpha=0.6, label='90% варијанса')
    ax.axhline(95, color='orange', linestyle='--', alpha=0.6, label='95% варијанса')
    ax.set_xlabel('Number of PCA components', fontsize=10)
    ax.set_ylabel('Cumulative variance (%)', fontsize=10)
    ax.set_title(f'{model_name} [{dataset_name}] — PCA Scree Plot',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'pca_variance_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_label_co_occurrence_in_embedding(model_name, dataset_name,
                                          emb_tsne, emb_umap,
                                          true_labels_arr, output_dir,
                                          label_cols=LABEL_COLS):
    pairs = list(combinations(range(len(label_cols)), 2))
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'{model_name} [{dataset_name}] — Label Co-occurrence (UMAP)',
                 fontsize=12, fontweight='bold')

    pair_colors = {
        'both': '#8e44ad',
        'only_a': '#e74c3c',
        'only_b': '#3498db',
        'neither': 'lightgray',
    }

    for ax, (i, j) in zip(axes, pairs):
        la, lb = label_cols[i], label_cols[j]
        a = true_labels_arr[:, i]
        b = true_labels_arr[:, j]

        for label_key, mask in [
            ('neither', (a == 0) & (b == 0)),
            ('only_a', (a == 1) & (b == 0)),
            ('only_b', (a == 0) & (b == 1)),
            ('both', (a == 1) & (b == 1)),
        ]:
            if mask.sum() == 0:
                continue
            legend_lbl = {
                'both': f'both (n={mask.sum()})',
                'only_a': f'only {la} (n={mask.sum()})',
                'only_b': f'only {lb} (n={mask.sum()})',
                'neither': f'neither (n={mask.sum()})',
            }[label_key]
            ax.scatter(emb_umap[mask, 0], emb_umap[mask, 1],
                       c=pair_colors[label_key], alpha=0.55, s=16,
                       linewidths=0, label=legend_lbl)

        ax.set_title(f'{la} vs {lb}', fontsize=10, fontweight='bold')
        ax.set_xlabel('UMAP Dim 1', fontsize=8)
        ax.set_ylabel('UMAP Dim 2', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    tag = f'{model_name.lower().replace("-", "_")}_{dataset_name.lower()}'
    path = os.path.join(output_dir, f'label_cooccurrence_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_mean_silhouette_summary(summary_rows, output_dir):
    df = pd.DataFrame(summary_rows)
    models = df['model'].unique()
    datasets = df['dataset'].unique()

    x = np.arange(len(models))
    width = 0.35
    ds_colors = {'dbAMP': '#3498db', 'DRAMP': '#e74c3c'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mean Silhouette Score (per-label average) per model',
                 fontsize=13, fontweight='bold')

    for ax, proj in zip(axes, ['tsne', 'umap']):
        for offset, ds in zip([-width / 2, width / 2], datasets):
            vals = []
            for m in models:
                row = df[(df['model'] == m) & (df['dataset'] == ds)]
                val = (float(row[f'{proj}_mean_sil_true'].values[0])
                       if len(row) and row[f'{proj}_mean_sil_true'].values[0] is not None
                       else 0.0)
                vals.append(val)
            bars = ax.bar(x + offset, vals, width,
                          label=ds, color=ds_colors.get(ds, 'gray'),
                          alpha=0.8, edgecolor='white')
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

        ax.set_title(f'{proj.upper()} mean silhouette', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel('Silhouette score')
        ax.legend(title='Dataset', fontsize=9)
        ax.set_ylim([-0.1, 1.0])
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'silhouette_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def run_model_dataset(model_name, dataset_name, df,
                      model_obj, tokenizer, model_type,
                      model_results, summary_rows, output_dir,
                      label_cols=LABEL_COLS):
    print(f'\n  [{dataset_name}]')
    true_labels_arr = df[label_cols].values.astype(int)

    embs, pred_probs, pred_binary = extract_embeddings(
        df, model_obj, tokenizer, model_type)

    emb_tsne, emb_umap, _, pca = reduce_embeddings(embs)

    mt = compute_multilabel_separation_metrics(emb_tsne, true_labels_arr, pred_binary, label_cols)
    mu = compute_multilabel_separation_metrics(emb_umap, true_labels_arr, pred_binary, label_cols)

    print(f'  t-SNE mean sil (true): {mt.get("mean_sil_true")}')
    print(f'  UMAP  mean sil (true): {mu.get("mean_sil_true")}')
    print(f'  Subset Accuracy: {mt.get("subset_accuracy_pct")}%')

    plot_single_model(model_name, dataset_name, emb_tsne, emb_umap,
                      true_labels_arr, pred_binary, mt, mu, output_dir, label_cols)
    plot_pca_variance(model_name, dataset_name, pca, output_dir)
    plot_label_co_occurrence_in_embedding(model_name, dataset_name,
                                          emb_tsne, emb_umap,
                                          true_labels_arr, output_dir, label_cols)

    key = (model_name, dataset_name)
    model_results[key] = dict(
        emb_tsne=emb_tsne, emb_umap=emb_umap,
        true_labels=true_labels_arr, pred_binary=pred_binary,
        metrics_tsne=mt, metrics_umap=mu
    )
    summary_rows.append({
        'model': model_name,
        'dataset': dataset_name,
        'tsne_mean_sil_true': mt.get('mean_sil_true'),
        'umap_mean_sil_true': mu.get('mean_sil_true'),
        'subset_accuracy_pct': mt.get('subset_accuracy_pct'),
        **{f'tsne_sil_{l}': mt.get(f'sil_true_{l}') for l in label_cols},
        **{f'umap_sil_{l}': mu.get(f'sil_true_{l}') for l in label_cols},
        **{f'acc_{l}': mt.get(f'acc_{l}') for l in label_cols},
    })

    del embs
    return model_results, summary_rows


if __name__ == '__main__':
    print('=' * 70)
    print('EMBEDDING VISUALIZATION — ProtBERT · ESM-2 · ProtT5  [MULTILABEL]')
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
        print(f'\nReading: {csv_path}')
        df = pd.read_csv(csv_path)
        if MAX_SAMPLES:
            df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=RANDOM_STATE)
        df = df.reset_index(drop=True)
        print(f'  {name}: {len(df)} sequences')
        for lbl in LABEL_COLS:
            print(f'    {lbl}: {df[lbl].sum()} positive')
        datasets[name] = df

    model_results = {}  # {(model_name, dataset_name): {...}}
    summary_rows = []

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

    for model_name, model_fn, tok_fn, mtype in model_configs:
        print(f'\n{"─" * 60}')
        print(f'── {model_name}')
        print(f'{"─" * 60}')

        tokenizer = tok_fn()

        for dataset_name, df in datasets.items():
            pth = pth_map[model_name][dataset_name]
            model_obj = load_model(model_fn(), pth, f'{model_name} [{dataset_name}]')

            model_results, summary_rows = run_model_dataset(
                model_name, dataset_name, df,
                model_obj, tokenizer, mtype,
                model_results, summary_rows, OUTPUT_DIR
            )

            del model_obj
            torch.cuda.empty_cache()
            gc.collect()

    print('\n── COMPARISON — all three models ──────────────────────────────────────')
    for dataset_name in datasets:
        plot_three_model_comparison(model_results, dataset_name, OUTPUT_DIR)

    plot_mean_silhouette_summary(summary_rows, OUTPUT_DIR)

    print('\n── SUMMARY TABLE ──────────────────────────────────────────────────')
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, 'embedding_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(summary_df.to_string(index=False))
    print(f'\nSaved: {csv_path}')

    print('\n' + '=' * 70)
    print(f'End  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 70)
