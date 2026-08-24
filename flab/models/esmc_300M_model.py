import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-300M")
model = AutoModelForMaskedLM.from_pretrained("biohub/ESMC-300M").to(device)
model.eval()


def esmc_300M_score(seq: str) -> float:
    """
    Compute pseudo-perplexity using a single forward pass.
    """
    inputs = tokenizer(seq, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    logits = logits[:, 1:-1, :]
    labels = inputs["input_ids"][:, 1:-1]

    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    nll = -token_log_probs.mean()

    return float(torch.exp(nll).cpu())
