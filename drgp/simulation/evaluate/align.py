"""Factor alignment: match inferred programs to planted ones before scoring recovery."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _l2_normalize(x: np.ndarray, axis: int = 0) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def hungarian_match(beta_hat: np.ndarray, beta_star: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Assign each truth column to one inferred column minimizing 1 - cos.

    beta_hat: (p, K_hat); beta_star: (p, L_truth). Returns (assign of len L_truth,
    cost matrix L_truth × K_hat)."""
    bh = _l2_normalize(beta_hat, axis=0)
    bs = _l2_normalize(beta_star, axis=0)
    cost = 1.0 - bs.T @ bh                     # (L_truth, K_hat)
    row, col = linear_sum_assignment(cost)
    assign = np.full(cost.shape[0], -1, dtype=np.int64)
    assign[row] = col
    return assign, cost


def recovery_cosine(beta_hat: np.ndarray, beta_star: np.ndarray,
                    assign: np.ndarray) -> np.ndarray:
    bh = _l2_normalize(beta_hat, axis=0)
    bs = _l2_normalize(beta_star, axis=0)
    return np.array([float(bs[:, l] @ bh[:, assign[l]]) for l in range(len(assign))])
