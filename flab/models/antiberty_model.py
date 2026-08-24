import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from antiberty import AntiBERTyRunner

antiberty = AntiBERTyRunner()


def antiberty_pseudoppl(seq):
    pll = antiberty.pseudo_log_likelihood(
        [seq],
        batch_size=16
    ).item()

    return math.exp(-pll)
