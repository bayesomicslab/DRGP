"""Program-recovery metrics.

support_auprc is the headline metric: rank genes by |beta_hat| on the matched factor and score
average precision against true carrier membership. It depends only on the gene ranking, so it is
comparable across methods and sign-agnostic (a down-regulated program is not penalised).
Chance level is |S_l| / p.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def recovery_jaccard_oracle(beta_hat: np.ndarray, S_star: list[np.ndarray],
                            assign: np.ndarray) -> list[float]:
    """Oracle top-|S*_l| cut on the matched factor — same rule for DRGP and baselines."""
    out = []
    for l, S_l in enumerate(S_star):
        k = assign[l]
        size = len(S_l)
        top = np.argsort(np.abs(beta_hat[:, k]))[-size:]
        inter = len(set(top.tolist()) & set(S_l.tolist()))
        union = size + size - inter
        out.append(inter / union if union > 0 else 0.0)
    return out


def support_auprc(beta_hat: np.ndarray, S_star: list[np.ndarray],
                           assign: np.ndarray) -> list[float]:
    """Scale-free membership recovery: rank genes by |beta_hat[:, assign[l]]|, label = in S_l,
    score by average precision (AUPRC). Unlike the magnitude cosine -- which is dominated by the
    coefficient-of-variation of the loadings on the carrier support and so rewards a flat profile
    rather than recovery -- AUPRC depends only on the gene ranking and is
    comparable across methods. Chance level is the support prevalence |S_l| / p."""
    p = beta_hat.shape[0]
    out = []
    for l, S_l in enumerate(S_star):
        lab = np.zeros(p, dtype=int)
        lab[np.asarray(S_l)] = 1
        score = np.abs(beta_hat[:, assign[l]])
        out.append(float(average_precision_score(lab, score)) if lab.any() else float("nan"))
    return out


recovery_support_auprc = support_auprc
