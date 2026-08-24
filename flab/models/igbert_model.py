import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Exscientia/IgBert").to(device)
model.eval()

# Special token IDs to exclude from perplexity computation
_SPECIAL_IDS = {
    tokenizer.cls_token_id,
    tokenizer.sep_token_id,
    tokenizer.pad_token_id,
    tokenizer.unk_token_id,
}


def igbert_paired_score(heavy_seq: str, light_seq: str = None) -> float:
    """
    Compute pseudo-perplexity for IgBERT using a single forward pass.

    Input format follows the IgBERT convention:
      - Paired:     "V Q L ... [SEP] E V V ..."
      - Heavy-only: "V Q L ..."

    Special tokens (CLS, SEP, PAD, UNK) are excluded from the perplexity
    calculation so only amino acid positions contribute to the score.
    """
    if light_seq:
        input_str = ' '.join(heavy_seq) + ' [SEP] ' + ' '.join(light_seq)
    else:
        input_str = ' '.join(heavy_seq)

    tokens = tokenizer(
        input_str,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    input_ids = tokens["input_ids"][0]  # [seq_len]

    aa_mask = torch.tensor(
        [tid.item() not in _SPECIAL_IDS for tid in input_ids],
        dtype=torch.bool,
        device=device,
    )

    logits_aa = logits[0][aa_mask]
    labels_aa = input_ids[aa_mask]

    log_probs = F.log_softmax(logits_aa, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels_aa.unsqueeze(-1),
    ).squeeze(-1)

    nll = -token_log_probs.mean()
    return float(torch.exp(nll).cpu())
