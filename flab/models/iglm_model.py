import math
from iglm import IgLM

iglm = IgLM()


def iglm_ppl(seq, chain_token):
    """
    Calculate IgLM perplexity of a given sequence.
    chain_token: "[HEAVY]" or "[LIGHT]"
    """
    log_likelihood = iglm.log_likelihood(seq, chain_token, "[HUMAN]")
    return math.exp(-log_likelihood)
