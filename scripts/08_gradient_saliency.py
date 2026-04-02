import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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

USE_SAVED_WEIGHTS = True
OUTPUT_DIR = '../results/saliency_viz_dramp'

LABELS = ['antimicrobial', 'antifungal', 'antiviral', 'anticancer']
NUM_LABELS = len(LABELS)

LABEL_COLORS = {
    'antimicrobial': '#2ecc71',
    'antifungal': '#e67e22',
    'antiviral': '#3498db',
    'anticancer': '#e74c3c',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_SEQUENCES = {
    'AMP_1 (magainin)': 'GIGKFLHSAKKFGKAFVGEIMNS',
    'AMP_2 (defensin)': 'ACYCRIPACIAGERRYGTCIYQGRLWAFCC',
    'nonAMP_1': 'MKLLFAIPVAVALAAGVQPQDAPSVAQKLEE',
    'nonAMP_2': 'GASVVDLNKLTQPDQSAGAKNLGKISQTLK',
}

SPECIAL_TOKENS = {
    '[CLS]', '[SEP]', '<cls>', '<eos>', '<pad>',
    '<unk>', '</s>', '<s>', '▁',
}


class ProtBERTMultiLabelClassifier(nn.Module):
    def __init__(self, n_classes=NUM_LABELS, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class ESM2MultiLabelClassifier(nn.Module):
    def __init__(self, model_name='facebook/esm2_t6_8M_UR50D', n_classes=NUM_LABELS, dropout=0.3):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.esm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class ProtT5MultiLabelClassifier(nn.Module):
    def __init__(self, model_name='Rostlab/prot_t5_xl_half_uniref50-enc',
                 n_classes=NUM_LABELS, dropout=0.3, freeze_t5=True):
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
        mean_emb = torch.sum(embeddings * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        return self.classifier(self.dropout(mean_emb))


def tokenize_sequence(sequence, tokenizer, model_type):
    """Tokenize a peptide sequence according to model-specific formatting requirements."""
    if model_type == 'bert':
        formatted = ' '.join(list(sequence))
    elif model_type == 't5':
        seq = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
        formatted = f"<AA2fold> {' '.join(list(seq))}"
    else:
        formatted = sequence

    encoding = tokenizer(
        formatted,
        return_tensors='pt',
        add_special_tokens=True,
        max_length=256,
        truncation=True,
    ).to(device)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    return encoding, tokens


def compute_gradient_saliency(sequence, model, tokenizer, model_type):
    """
    Compute per-residue gradient saliency scores for each output label.

    For each label, runs a backward pass from the sigmoid-activated logit through
    the input embedding layer, then reduces the gradient to a scalar per token
    by taking the L2 norm over the embedding dimension. Returns saliency arrays
    of shape (NUM_LABELS, seq_len) alongside the decoded token list and
    predicted binary label vector.
    """
    encoding, tokens = tokenize_sequence(sequence, tokenizer, model_type)

    if model_type == 'bert':
        embed_layer = model.bert.embeddings.word_embeddings
    elif model_type == 'esm':
        embed_layer = model.esm.embeddings.word_embeddings
    else:
        embed_layer = model.t5.shared

    input_embeds = embed_layer(encoding['input_ids']).detach().requires_grad_(True)

    if model_type == 'bert':
        outputs_raw = model.bert(
            inputs_embeds=input_embeds,
            attention_mask=encoding['attention_mask'],
        )
        cls_rep = outputs_raw.last_hidden_state[:, 0, :]
        logits = model.classifier(model.dropout(cls_rep))
    elif model_type == 'esm':
        esm = model.esm
        # Temporarily disable token_dropout to avoid the masked_fill bug
        # (it requires input_ids which we can't pass alongside inputs_embeds)
        original_token_dropout = esm.embeddings.token_dropout
        esm.embeddings.token_dropout = False

        outputs_raw = esm(
            inputs_embeds=input_embeds,
            attention_mask=encoding['attention_mask'],
        )

        esm.embeddings.token_dropout = original_token_dropout
        cls_rep = outputs_raw.last_hidden_state[:, 0, :]
        logits = model.classifier(model.dropout(cls_rep))
    else:
        outputs_raw = model.t5(
            inputs_embeds=input_embeds,
            attention_mask=encoding['attention_mask'],
        )
        embeddings = outputs_raw.last_hidden_state
        mask_exp = encoding['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        mean_emb = torch.sum(embeddings * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        logits = model.classifier(model.dropout(mean_emb))

    probs = torch.sigmoid(logits)
    pred_labels = (probs > 0.5).squeeze(0).cpu().numpy()

    saliency_all = np.zeros((NUM_LABELS, input_embeds.shape[1]))

    for label_idx in range(NUM_LABELS):
        model.zero_grad()
        if input_embeds.grad is not None:
            input_embeds.grad.zero_()
        scalar = probs[0, label_idx]
        scalar.backward(retain_graph=(label_idx < NUM_LABELS - 1))
        grad = input_embeds.grad.detach().cpu().numpy()[0]
        saliency_all[label_idx] = np.linalg.norm(grad, axis=-1)

    return saliency_all, tokens, pred_labels, probs.detach().cpu().numpy()[0]


def filter_tokens(tokens, saliency_row):
    """Remove special tokens and return clean (token, saliency) pairs."""
    keep = [
        i for i, t in enumerate(tokens)
        if t not in SPECIAL_TOKENS and len(t.strip()) > 0
    ]
    clean_tokens = [tokens[i].replace('▁', '') for i in keep]
    clean_saliency = saliency_row[keep]
    return clean_tokens, clean_saliency


def normalize_saliency(saliency):
    """Min-max normalize saliency scores to [0, 1]."""
    mn, mx = saliency.min(), saliency.max()
    if mx - mn < 1e-9:
        return np.zeros_like(saliency)
    return (saliency - mn) / (mx - mn)


def bar_colors_from_saliency(saliency, label_name):
    """Assign bar colors based on saliency percentile thresholds for a given label."""
    base = LABEL_COLORS[label_name]
    return [
        base if v > np.percentile(saliency, 75)
        else '#95a5a6' if v < np.percentile(saliency, 25)
        else '#bdc3c7'
        for v in saliency
    ]


def safe_name(s):
    """Convert a sequence name to a filesystem-safe string."""
    return s[:15].replace(' ', '_').replace('(', '').replace(')', '')


def plot_per_label_saliency(saliency_all, tokens, pred_labels, probs,
                            seq_name, sequence, model_name, output_dir):
    """
    Plot a 2x2 grid of per-label saliency bar charts for a single sequence.

    Each subplot corresponds to one of the four activity labels and shows
    normalized gradient saliency per amino acid position. Predicted-positive
    labels are indicated in the subplot title alongside sigmoid probability.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    pred_str = ', '.join([LABELS[i] for i, p in enumerate(pred_labels) if p]) or 'none'
    fig.suptitle(
        f'{model_name}  |  {seq_name}  |  Predicted: [{pred_str}]\n{sequence}',
        fontsize=13, fontweight='bold', y=0.99,
    )

    for idx, label_name in enumerate(LABELS):
        ax = axes[idx]
        clean_tok, sal = filter_tokens(tokens, saliency_all[idx])
        sal_norm = normalize_saliency(sal)

        colors = bar_colors_from_saliency(sal_norm, label_name)
        ax.bar(range(len(clean_tok)), sal_norm, color=colors, edgecolor='none')
        ax.set_xticks(range(len(clean_tok)))
        ax.set_xticklabels(clean_tok, fontsize=8)
        ax.set_ylabel('Normalized saliency')
        ax.set_xlabel('Amino acid')
        ax.grid(axis='y', alpha=0.3)

        predicted = pred_labels[idx]
        prob_val = probs[idx]
        title_suffix = f'predicted ({prob_val:.2f})' if predicted else f'not predicted ({prob_val:.2f})'
        ax.set_title(
            f'{label_name}  —  {title_suffix}',
            fontsize=10, fontweight='bold',
            color=LABEL_COLORS[label_name],
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(
        output_dir,
        f'saliency_{model_name.lower().replace("-", "_")}_{safe_name(seq_name)}.png',
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_label_overlap_heatmap(saliency_all, tokens, seq_name, sequence,
                               model_name, output_dir):
    """
    Plot a heatmap of normalized saliency scores across all four labels.

    Rows are labels, columns are amino acid positions. Provides a compact
    view of which residues are jointly important across multiple activity classes.
    """
    clean_tok, _ = filter_tokens(tokens, saliency_all[0])
    matrix = np.zeros((NUM_LABELS, len(clean_tok)))

    for i in range(NUM_LABELS):
        _, sal = filter_tokens(tokens, saliency_all[i])
        matrix[i] = normalize_saliency(sal)

    fig, ax = plt.subplots(figsize=(max(12, len(clean_tok) * 0.55), 4))
    sns.heatmap(
        matrix,
        xticklabels=clean_tok,
        yticklabels=LABELS,
        ax=ax,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        linewidths=0.4,
        linecolor='#eeeeee',
        cbar_kws={'label': 'Normalized saliency'},
    )
    ax.set_title(
        f'{model_name}  |  {seq_name}  —  Saliency across labels\n{sequence}',
        fontsize=11, fontweight='bold',
    )
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=9)

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f'saliency_heatmap_{model_name.lower().replace("-", "_")}_{safe_name(seq_name)}.png',
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_three_model_comparison(seq_name, sequence, results, label_name, output_dir):
    """
    Compare per-position saliency for a single label across all three models.

    Produces a 1x3 subplot figure where each panel shows normalized gradient
    saliency for the given label from ESM-2, ProtBERT, and ProtT5 respectively.
    """
    models = ['ESM-2', 'ProtBERT', 'ProtT5']
    label_idx = LABELS.index(label_name)

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.suptitle(
        f'ESM-2 vs ProtBERT vs ProtT5  |  Label: {label_name}  |  {seq_name}\n{sequence}',
        fontsize=12, fontweight='bold',
    )

    for ax, model_name in zip(axes, models):
        data = results.get(model_name)
        if data is None:
            ax.set_visible(False)
            continue

        saliency_all, tokens, pred_labels, probs = data
        clean_tok, sal = filter_tokens(tokens, saliency_all[label_idx])
        sal_norm = normalize_saliency(sal)

        colors = bar_colors_from_saliency(sal_norm, label_name)
        ax.bar(range(len(clean_tok)), sal_norm, color=colors)
        ax.set_xticks(range(len(clean_tok)))
        ax.set_xticklabels(clean_tok, fontsize=8)
        predicted = pred_labels[label_idx]
        prob_val = probs[label_idx]
        ax.set_title(
            f'{model_name}\n{label_name} ({prob_val:.2f}) {"predicted" if predicted else "not predicted"}',
            fontsize=11,
        )
        ax.set_ylabel('Normalized saliency')
        ax.set_xlabel('Amino acid')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f'comparison_{label_name}_{safe_name(seq_name)}.png',
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def compute_saliency_entropy(saliency_all):
    """
    Compute mean Shannon entropy of normalized saliency distributions across all labels.

    Higher entropy indicates more diffuse saliency (no dominant position);
    lower entropy indicates the model concentrates gradient signal on fewer residues.
    """
    eps = 1e-9
    entropies = []
    for i in range(NUM_LABELS):
        sal = saliency_all[i]
        sal_norm = sal / (sal.sum() + eps)
        entropies.append(float(-np.sum(sal_norm * np.log(sal_norm + eps))))
    return float(np.mean(entropies))


def load_model(model_obj, pth_path, label):
    """Load saved weights into a model if the checkpoint exists, then set to eval mode."""
    if USE_SAVED_WEIGHTS and os.path.exists(pth_path):
        model_obj.load_state_dict(torch.load(pth_path, map_location=device))
        print(f'Weights: {pth_path}')
    else:
        print(f'WARNING: {pth_path} does not exist — using pretrained weights only')
    model_obj.eval().to(device)
    print(f'{label} is ready.')
    return model_obj


if __name__ == '__main__':
    print('GRADIENT SALIENCY VISUALIZATION — ProtBERT · ESM-2 · ProtT5')
    print(f'Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {name: {} for name in TEST_SEQUENCES}

    print('\n── ESM-2 ──────────────────────────────────────────────────────────────────')
    esm2_tok = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    esm2_mdl = load_model(ESM2MultiLabelClassifier(), ESM2_PTH, 'ESM-2')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        sal, tok, pred, probs = compute_gradient_saliency(seq, esm2_mdl, esm2_tok, 'esm')
        all_results[name]['ESM-2'] = (sal, tok, pred, probs)
        predicted_labels = [LABELS[i] for i, p in enumerate(pred) if p]
        print(f'  → predicted: {predicted_labels if predicted_labels else ["none"]}')
        print(f'  → probs: { {l: round(float(probs[i]), 3) for i, l in enumerate(LABELS)} }')
        plot_per_label_saliency(sal, tok, pred, probs, name, seq, 'ESM-2', OUTPUT_DIR)
        plot_label_overlap_heatmap(sal, tok, name, seq, 'ESM-2', OUTPUT_DIR)

    del esm2_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── ProtBERT ────────────────────────────────────────────────────────────────')
    pb_tok = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
    pb_mdl = load_model(ProtBERTMultiLabelClassifier(), PROTBERT_PTH, 'ProtBERT')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        sal, tok, pred, probs = compute_gradient_saliency(seq, pb_mdl, pb_tok, 'bert')
        all_results[name]['ProtBERT'] = (sal, tok, pred, probs)
        predicted_labels = [LABELS[i] for i, p in enumerate(pred) if p]
        print(f'  → predicted: {predicted_labels if predicted_labels else ["none"]}')
        print(f'  → probs: { {l: round(float(probs[i]), 3) for i, l in enumerate(LABELS)} }')
        plot_per_label_saliency(sal, tok, pred, probs, name, seq, 'ProtBERT', OUTPUT_DIR)
        plot_label_overlap_heatmap(sal, tok, name, seq, 'ProtBERT', OUTPUT_DIR)

    del pb_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── ProtT5 ──────────────────────────────────────────────────────────────────')
    t5_tok = T5Tokenizer.from_pretrained(
        'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False
    )
    t5_mdl = load_model(ProtT5MultiLabelClassifier(freeze_t5=True), PROTT5_PTH, 'ProtT5')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        sal, tok, pred, probs = compute_gradient_saliency(seq, t5_mdl, t5_tok, 't5')
        all_results[name]['ProtT5'] = (sal, tok, pred, probs)
        predicted_labels = [LABELS[i] for i, p in enumerate(pred) if p]
        print(f'  → predicted: {predicted_labels if predicted_labels else ["none"]}')
        print(f'  → probs: { {l: round(float(probs[i]), 3) for i, l in enumerate(LABELS)} }')
        plot_per_label_saliency(sal, tok, pred, probs, name, seq, 'ProtT5', OUTPUT_DIR)
        plot_label_overlap_heatmap(sal, tok, name, seq, 'ProtT5', OUTPUT_DIR)

    del t5_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── COMPARISON — per label across models ───────────────────────────────────')
    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        for label_name in LABELS:
            plot_three_model_comparison(name, seq, all_results[name], label_name, OUTPUT_DIR)

    print('\n── SUMMARY TABLE ──────────────────────────────────────────────────────────────────')
    rows = []
    for name, seq in TEST_SEQUENCES.items():
        row = {'sequence_name': name, 'sequence': seq}
        for mdl_name, data in all_results[name].items():
            sal_all, _, pred, probs = data
            for i, label_name in enumerate(LABELS):
                row[f'{mdl_name}_{label_name}_prob'] = round(float(probs[i]), 4)
                row[f'{mdl_name}_{label_name}_predicted'] = bool(pred[i])
            row[f'{mdl_name}_saliency_entropy'] = round(compute_saliency_entropy(sal_all), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'saliency_summary.csv')
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f'\nSaved at: {csv_path}')

    print('\nDONE!')
    print(f'End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
