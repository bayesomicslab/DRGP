"""Metric computation and the DRGP output-artifact contract (extracted from utils.py)."""

import os
import gzip
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
    y_proba : ndarray, optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict
        Dictionary of metric names to values.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        confusion_matrix
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            # Ensure arrays are 1D for consistent handling
            y_true_flat = np.asarray(y_true).ravel()
            y_proba_flat = np.asarray(y_proba).ravel()
            metrics['auc'] = roc_auc_score(y_true_flat, y_proba_flat)
            metrics['average_precision'] = average_precision_score(y_true_flat, y_proba_flat)
        except ValueError:
            # Handle case where only one class is present
            metrics['auc'] = float('nan')
            metrics['average_precision'] = float('nan')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])

    return metrics


def get_top_genes_per_program(
    model: Any,
    gene_list: List[str],
    n_top: int = 10,
    threshold: float = 0.5,
    feature_type: str = 'gene'
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Get top genes/pathways for each gene program.

    Parameters
    ----------
    model : VI
        Trained model with E_beta attribute.
    gene_list : list of str
        List of gene/pathway names corresponding to beta rows.
    n_top : int, default=10
        Number of top genes/pathways to return per program.
    threshold : float, default=0.5
        Spike-and-slab threshold for considering genes/pathways active.
    feature_type : str, default='gene'
        Type of features ('gene' or 'pathway').

    Returns
    -------
    dict
        Dictionary mapping program names to lists of (gene/pathway, loading) tuples.
    """
    results = {}

    for k in range(model.K):
        program_name = f"GP{k+1}"

        # Get gene loadings for this program
        loadings = model.E_beta[:, k].copy()

        # Get top genes by E[beta] loading
        top_indices = np.argsort(loadings)[::-1][:n_top]
        top_genes = [(gene_list[i], float(loadings[i])) for i in top_indices]

        results[program_name] = top_genes

    return results


def print_model_summary(model: Any, gene_list: Optional[List[str]] = None) -> None:
    """
    Print a summary of the trained model.

    Parameters
    ----------
    model : VI
        Trained model instance.
    gene_list : list of str, optional
        List of gene names for displaying top genes.
    """
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print(f"\nHyperparameters (scHPF):")
    print(f"  n_factors (K): {model.K}")
    print(f"  a: {model.a}")
    print(f"  c: {model.c}")
    print(f"  b_v: {model.b_v}")
    print(f"  sigma_gamma: {model.sigma_gamma}")

    if hasattr(model, 'seed_used_'):
        print(f"  random_state: {model.seed_used_} {'(random)' if model.seed_used_ is None else ''}")

    print(f"\nLearned Parameters:")
    print(f"  E[beta] shape: {model.E_beta.shape}")
    print(f"    range: [{model.E_beta.min():.4f}, {model.E_beta.max():.4f}]")
    print(f"    mean: {model.E_beta.mean():.4f}")

    print(f"  mu_v shape: {model.mu_v.shape}")
    print(f"    range: [{model.mu_v.min():.4f}, {model.mu_v.max():.4f}]")

    if hasattr(model, 'E_theta'):
        print(f"  E[theta] shape: {model.E_theta.shape}")
        print(f"    range: [{model.E_theta.min():.4f}, {model.E_theta.max():.4f}]")

    # Training info
    if hasattr(model, 'elbo_history_') and model.elbo_history_:
        print(f"\nTraining:")
        print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
        print(f"  Iterations: {model.elbo_history_[-1][0] + 1}")
    if hasattr(model, 'holl_history_') and model.holl_history_:
        print(f"  Best HO-LL: {max(entry[1] for entry in model.holl_history_):.4f}")

    # Top genes per program
    if gene_list is not None:
        top_genes = get_top_genes_per_program(model, gene_list, n_top=5)
        _mv0 = np.asarray(model.mu_v[0])
        if _mv0.ndim > 1:
            _mv0 = _mv0.mean(axis=0)
        most_influential = np.argmax(np.abs(_mv0)) if model.mu_v.size > 0 else 0
        print(f"\nTop 5 genes in most influential program (GP{most_influential + 1}):")
        for gene, loading in top_genes[f'GP{most_influential + 1}']:
            print(f"    {gene}: {loading:.4f}")

    print("=" * 60)


def plot_training_curves(model, save_dir, fname="training_curves.png"):
    """
    Generate ELBO and held-out LL convergence plots with component breakdowns.

    Produces a 1×N row (iter 0 excluded to show real dynamics).
    Each panel shows component breakdowns (Poisson, Regression, Bernoulli).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_elbo = hasattr(model, 'elbo_history_') and bool(model.elbo_history_)
    has_holl = hasattr(model, 'holl_history_') and bool(model.holl_history_)
    if not has_elbo and not has_holl:
        print("No training history available. Run fit() first.")
        return

    # ── Extract data ──
    elbo_data = {}
    if has_elbo:
        first = model.elbo_history_[0]
        elbo_data['iters'] = [e[0] for e in model.elbo_history_]
        elbo_data['elbo'] = [e[1] for e in model.elbo_history_]
        if len(first) >= 4:
            elbo_data['pois'] = [e[2] for e in model.elbo_history_]
            elbo_data['reg'] = [e[3] for e in model.elbo_history_]

    holl_data = {}
    if has_holl:
        first = model.holl_history_[0]
        holl_data['iters'] = [e[0] for e in model.holl_history_]
        holl_data['total'] = [e[1] for e in model.holl_history_]
        if len(first) >= 3:
            holl_data['pois'] = [e[2] for e in model.holl_history_]
        if len(first) >= 4:
            holl_data['reg'] = [e[3] for e in model.holl_history_]
        if len(first) >= 5:
            holl_data['bern'] = [e[4] for e in model.holl_history_]

    n_cols = int(has_elbo) + int(has_holl)

    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5),
                             squeeze=False)

    def _plot_elbo(ax, iters, data, title_suffix=''):
        ax.plot(iters, data['elbo'], 'k-', lw=1.5, marker='.', ms=3,
                label='ELBO (total)')
        if 'pois' in data:
            ax.plot(iters, data['pois'], 'steelblue', lw=1, ls='--',
                    marker='.', ms=2, label='Poisson LL')
            ax2 = ax.twinx()
            ax2.plot(iters, data['reg'], 'coral', lw=1, ls='--',
                     marker='.', ms=2, label='Regression LL')
            ax2.set_ylabel('Regression LL', fontsize=9, color='coral')
            ax2.tick_params(axis='y', labelcolor='coral')
            ax2.legend(loc='center right', fontsize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.set_title(f'ELBO{title_suffix}')
        ax.legend(loc='upper left', fontsize=8)

    def _plot_holl(ax, iters, data, title_suffix=''):
        ax.plot(iters, data['total'], 'k-', lw=1.5, marker='.', ms=3,
                label='Total HO-LL')
        if 'pois' in data:
            ax.plot(iters, data['pois'], 'steelblue', lw=1, ls='--',
                    marker='.', ms=2, label='HO Poisson')
        if 'reg' in data or 'bern' in data:
            ax2 = ax.twinx()
            if 'reg' in data:
                ax2.plot(iters, data['reg'], 'coral', lw=1, ls='--',
                         marker='.', ms=2, label='HO Reg')
            if 'bern' in data:
                ax2.plot(iters, data['bern'], 'mediumpurple', lw=1, ls=':',
                         marker='.', ms=2, label='HO Bernoulli (true)')
            ax2.set_ylabel('Regression LL / sample', fontsize=9)
            ax2.legend(loc='center right', fontsize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Held-out LL / sample')
        ax.set_title(f'Held-out Log-Likelihood{title_suffix}')
        ax.legend(loc='upper left', fontsize=8)

    # ── Skip first point (burn-in) to reveal dynamics ──
    col = 0
    if has_elbo:
        zoomed = {k: v[1:] for k, v in elbo_data.items()} \
                 if len(elbo_data['iters']) > 2 else elbo_data
        _plot_elbo(axes[0, col], zoomed['iters'],
                   {k: zoomed[k] for k in zoomed if k != 'iters'},
                   ' (iter 0 excluded)')
        col += 1
    if has_holl:
        zoomed = {k: v[1:] for k, v in holl_data.items()} \
                 if len(holl_data['iters']) > 2 else holl_data
        _plot_holl(axes[0, col], zoomed['iters'],
                   {k: zoomed[k] for k in zoomed if k != 'iters'},
                   ' (iter 0 excluded)')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(str(save_dir), fname)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close(fig)


def plot_diagnostics(diagnostics, save_dir, fname="diagnostics.png"):
    """
    Generate diagnostic plots for model quality assessment.

    Plots (3x2 grid):
    1. Train theta L1 norm distribution over iterations
    2. zeta saturation and lambda(zeta) over iterations
    3. True validation logistic loss vs PG-CAVI bound
    4. eta vs gene total counts (mask consistency)
    5. E[eta] range over iterations (eta-beta collapse detector)
    6. E[beta] range over iterations

    Parameters
    ----------
    diagnostics : dict
        The model's diagnostics_ dictionary.
    save_dir : str or Path
        Directory to save figure.
    fname : str
        Filename for the saved figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    diag = diagnostics
    if diag is None:
        print("No diagnostics available. Run fit() first.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))

    # --- Plot 1: Train theta L1 norms ---
    ax = axes[0, 0]
    if diag['theta_l1_train']:
        iters, means, stds, mins, maxs = zip(*diag['theta_l1_train'])
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(iters, means, 'b-', label='train mean')
        ax.fill_between(iters, means - stds, means + stds, alpha=0.2, color='b')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||E[theta_i]||_1')
    ax.set_title('Train theta L1 norms')
    ax.legend()

    # --- Plot 2: (was zeta saturation; zeta removed in PG-CAVI — plot wbar instead) ---
    ax = axes[0, 1]
    if diag.get('wbar_stats'):
        iters, wmins, wmeds, wmaxs = zip(*diag['wbar_stats'])
        ax.plot(iters, wmeds, 'g-', label='wbar median')
        ax.fill_between(iters, wmins, wmaxs, alpha=0.15, color='g')
        ax.legend(loc='upper left')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('wbar')
    ax.set_title('wbar (PG posterior mean) over iterations')

    # --- Plot 3: True Bernoulli LL vs PG-CAVI supervised LL ---
    ax = axes[1, 0]
    if diag.get('true_val_ll') and diag.get('bound_val_ll'):
        iters_t, vals_t = zip(*diag['true_val_ll'])
        iters_j, vals_j = zip(*diag['bound_val_ll'])
        ax.plot(iters_t, vals_t, 'b-o', markersize=3, label='True Bernoulli LL')
        ax.plot(iters_j, vals_j, 'r-s', markersize=3, label='PG-CAVI bound LL')
        ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation LL / sample')
    ax.set_title('True logistic loss vs PG-CAVI bound')

    # --- Plot 4: eta vs gene total counts (mask consistency) ---
    ax = axes[1, 1]
    if diag['eta_vs_counts'] is not None:
        E_eta, gene_counts, active_mask = diag['eta_vs_counts']
        if active_mask is not None:
            n_active = active_mask.sum(axis=1)
            scatter = ax.scatter(gene_counts, E_eta, c=n_active,
                                 cmap='viridis', s=3, alpha=0.5)
            plt.colorbar(scatter, ax=ax, label='n_active_factors')
        else:
            ax.scatter(gene_counts, E_eta, s=3, alpha=0.5)
    ax.set_xlabel('Gene total counts')
    ax.set_ylabel('E[eta_j]')
    ax.set_title('eta vs gene counts (mask consistency)')
    ax.set_xscale('log')

    # --- Plot 5: E[eta] over iterations (eta-beta collapse detector) ---
    ax = axes[2, 0]
    if diag.get('eta_stats'):
        iters, emins, emeds, emaxs = zip(*diag['eta_stats'])
        ax.semilogy(iters, emeds, 'b-', label='median')
        ax.fill_between(iters, emins, emaxs, alpha=0.15, color='b')
        ax.semilogy(iters, emaxs, 'r--', alpha=0.5, label='max')
    bp_dp = diag.get('bp_dp')
    if bp_dp:
        ax.set_title(f'E[eta] over iterations  (bp={bp_dp[0]:.4f}, dp={bp_dp[1]:.4f})')
    else:
        ax.set_title('E[eta] over iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E[eta_j]')
    ax.legend()

    # --- Plot 6: E[beta] over iterations ---
    ax = axes[2, 1]
    if diag.get('beta_stats'):
        iters, bmins, bmeds, bmaxs = zip(*diag['beta_stats'])
        ax.semilogy(iters, bmeds, 'g-', label='median')
        ax.fill_between(iters, bmins, bmaxs, alpha=0.15, color='g')
        ax.semilogy(iters, bmaxs, 'r--', alpha=0.5, label='max')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E[beta_jk]')
    ax.set_title('E[beta] over iterations')
    ax.legend()

    # --- Plot 7: v convergence trace ---
    ax = axes[3, 0]
    if diag.get('v_stats'):
        data = list(zip(*diag['v_stats']))
        iters = data[0]
        vmins, vmeds, vmaxs, mean_abs = data[1], data[2], data[3], data[4]
        n_near_zero, n_large = data[5], data[6]
        ax.plot(iters, mean_abs, 'b-', label='mean |v|')
        ax.fill_between(iters, vmins, vmaxs, alpha=0.12, color='b', label='min–max')
        ax.plot(iters, vmeds, 'b--', alpha=0.5, label='median v')
        ax2 = ax.twinx()
        ax2.plot(iters, n_near_zero, 'g:', label='|v|<0.5')
        ax2.plot(iters, n_large, 'r:', label='|v|>5')
        ax2.set_ylabel('Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.legend(loc='right', fontsize=8)
        ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('v')
    ax.set_title('v weights convergence')

    # --- Plot 8: gamma convergence trace ---
    ax = axes[3, 1]
    if diag.get('gamma_stats') and len(diag['gamma_stats']) > 0:
        data = list(zip(*diag['gamma_stats']))
        iters = data[0]
        n_gamma = len(data) - 1
        labels = ['intercept'] + [f'aux_{j}' for j in range(n_gamma - 1)]
        for j in range(n_gamma):
            ax.plot(iters, data[j + 1], label=labels[j])
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('gamma')
    ax.set_title('gamma (covariate weights) convergence')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(str(save_dir), fname)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostics saved to {save_path}")
    plt.close(fig)


def save_results(
    model: Any,
    output_dir: Union[str, Path],
    gene_list: List[str],
    splits: Dict[str, List[str]],
    prefix: str = 'vi',
    save_model: bool = True,
    compress: bool = True,
    save_full_model: bool = False,
    feature_type: str = 'gene',
    optimal_threshold: Union[float, Dict[str, float]] = 0.5,
    program_names: Optional[List[str]] = None,
    mode: str = 'unmasked',
    label_columns: Optional[List[str]] = None,
    aux_columns: Optional[List[str]] = None,
    val_test_data: Optional[Dict[str, Any]] = None,
    cell_metadata: Optional[Any] = None,
) -> Dict[str, Path]:
    """
    Save VI model results to files.

    Parameters
    ----------
    model : VI
        Trained model instance.
    output_dir : str or Path
        Directory to save results.
    gene_list : list of str
        List of gene/pathway names.
    splits : dict
        Dictionary with train/val/test cell ID lists.
    prefix : str, default='vi'
        Prefix for output files.
    save_model : bool, default=True
        Whether to save the model (essential parameters only by default).
    compress : bool, default=True
        Whether to compress CSV files with gzip.
    save_full_model : bool, default=False
        If True, save entire model object. If False (default), save only
        essential parameters to reduce memory during save.
    feature_type : str, default='gene'
        Type of features ('gene' or 'pathway').
    optimal_threshold : float or dict, default=0.5
        Optimal classification threshold tuned on validation set.
    program_names : list of str, optional
        Custom names for programs/factors (e.g., pathway names). If None,
        defaults to GP1, GP2, etc.
    mode : str, default='unmasked'
        Model mode ('unmasked', 'masked', or 'pathway_init').
    label_columns : list of str, optional
        Names of label columns (e.g., ['CoVID-19 severity', 'Outcome']).
        Used for naming v_weight columns and gamma rows in output files.
    aux_columns : list of str, optional
        Names of auxiliary feature columns. Used for naming gamma weight
        columns in output files.
    val_test_data : dict, optional
        Dictionary with validation/test data for inferring theta:
        {'X_val': ..., 'X_aux_val': ..., 'X_test': ..., 'X_aux_test': ...}
    cell_metadata : DataFrame, optional
        DataFrame indexed by cell ID with metadata columns (e.g. majorType).
        If provided, metadata columns are prepended to theta DataFrames.

    Returns
    -------
    dict
        Dictionary mapping result types to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}
    ext = '.csv.gz' if compress else '.csv'
    compression = 'gzip' if compress else None

    # Save model - either full or essential parameters only
    if save_model:
        if save_full_model:
            # Full model pickle (can be very large)
            model_path = output_dir / f'{prefix}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            saved_files['model'] = model_path
            print(f"Saved full model to {model_path}")
        else:
            # Save only essential parameters (memory-efficient)
            essential_params = {
                # Hyperparameters
                'n_factors': model.K,
                'a': model.a,
                'c': model.c,
                'b_v': model.b_v,
                'sigma_gamma': model.sigma_gamma,
                # Global parameters (needed for inference)
                'E_beta': model.E_beta,
                'E_log_beta': model.E_log_beta,
                # Classification weights
                'mu_v': model.mu_v,
                'sigma_v_diag': np.array(model.sigma_v_diag),
                'mu_gamma': model.mu_gamma,
                'Sigma_gamma': np.array(model.Sigma_gamma),
                # Dimensions
                'n': model.n,
                'p': model.p,
                'p_aux': model.p_aux,
            }

            # Spike-and-slab inclusion probabilities
            if hasattr(model, 'r_beta') and model.r_beta is not None:
                essential_params['r_beta'] = np.asarray(model.r_beta)

            # Optional: include training history
            if hasattr(model, 'elbo_history_'):
                essential_params['elbo_history'] = model.elbo_history_
            if hasattr(model, 'poisson_ll_history_'):
                essential_params['poisson_ll_history'] = model.poisson_ll_history_
            if hasattr(model, 'regression_ll_history_'):
                essential_params['regression_ll_history'] = model.regression_ll_history_
            if hasattr(model, 'training_time_'):
                essential_params['training_time'] = model.training_time_

            model_path = output_dir / f'{prefix}_model_params.npz'
            np.savez_compressed(model_path, **essential_params)
            saved_files['model_params'] = model_path
            print(f"Saved essential model parameters to {model_path}")

    # Save gene/pathway programs (beta matrix)
    # Use custom program names if provided (e.g., pathway names)
    if program_names is not None:
        prog_labels = program_names
    else:
        prog_labels = [f"GP{k+1}" for k in range(model.K)]

    program_label = f"{feature_type}_program" if feature_type == 'pathway' else "gene_program"
    beta_df = pd.DataFrame(
        model.E_beta.T,  # Transpose: programs x genes/pathways
        index=prog_labels,
        columns=gene_list
    )

    # Add classification weights if available
    if hasattr(model, 'mu_v'):
        n_outcomes = model.mu_v.shape[0]
        for k in range(n_outcomes):
            lname = label_columns[k] if label_columns is not None and k < len(label_columns) else f'class{k}'
            v_k = np.asarray(model.mu_v[k])
            beta_df.insert(0, f'v_weight_{lname}', v_k)

    beta_path = output_dir / f'{prefix}_{feature_type}_programs{ext}'
    beta_df.to_csv(beta_path, compression=compression)
    saved_files[f'{feature_type}_programs'] = beta_path
    print(f"Saved {feature_type} programs to {beta_path}")

    # Save posterior inclusion probabilities (r_beta) for spike-and-slab models
    if hasattr(model, 'r_beta') and model.r_beta is not None:
        r_beta_arr = np.asarray(model.r_beta)  # handles CuPy arrays too
        r_beta_df = pd.DataFrame(
            r_beta_arr.T,  # Transpose to programs x genes (same layout as beta_df)
            index=prog_labels,
            columns=gene_list
        )
        r_beta_path = output_dir / f'{prefix}_r_beta{ext}'
        r_beta_df.to_csv(r_beta_path, compression=compression)
        saved_files['r_beta'] = r_beta_path
        print(f"Saved r_beta (posterior inclusion probs) to {r_beta_path}")

    # --- Helper to add cell metadata columns to a theta DataFrame ---
    def _add_cell_metadata(theta_df):
        if cell_metadata is not None:
            # Match on index (cell IDs)
            common_ids = theta_df.index.intersection(cell_metadata.index)
            if len(common_ids) > 0:
                for col in cell_metadata.columns:
                    theta_df.insert(0, col, cell_metadata.reindex(theta_df.index)[col])
        return theta_df

    # Save training theta
    if hasattr(model, 'E_theta'):
        theta_train_df = pd.DataFrame(
            model.E_theta,
            index=splits['train'],
            columns=prog_labels
        )
        theta_train_df.index.name = 'cell_id'
        _add_cell_metadata(theta_train_df)
        theta_path = output_dir / f'{prefix}_theta_train{ext}'
        theta_train_df.to_csv(theta_path, compression=compression)
        saved_files['theta_train'] = theta_path
        print(f"Saved training theta to {theta_path}")

    # Save validation and test theta (inferred with frozen model parameters)
    if val_test_data is not None:
        for split_name in ('val', 'test'):
            X_key = f'X_{split_name}'
            X_aux_key = f'X_aux_{split_name}'
            if X_key not in val_test_data or split_name not in splits:
                continue
            X_split = val_test_data[X_key]
            X_aux_split = val_test_data.get(X_aux_key)
            result = model.transform(X_split, X_aux_new=X_aux_split)
            theta_split_df = pd.DataFrame(
                result['E_theta'],
                index=splits[split_name],
                columns=prog_labels
            )
            theta_split_df.index.name = 'cell_id'
            _add_cell_metadata(theta_split_df)
            theta_split_path = output_dir / f'{prefix}_theta_{split_name}{ext}'
            theta_split_df.to_csv(theta_split_path, compression=compression)
            saved_files[f'theta_{split_name}'] = theta_split_path
            print(f"Saved {split_name} theta to {theta_split_path}")

    # Save gamma (auxiliary feature weights) if available
    if hasattr(model, 'mu_gamma') and model.mu_gamma.size > 0:
        # mu_gamma shape: (kappa, p_aux) where p_aux may include a
        # prepended intercept column (col 0) added by _prepend_intercept.
        # Build column names: "intercept" + real aux column names.
        has_intercept = getattr(model, 'use_intercept', False)
        n_gamma_cols = model.mu_gamma.shape[1]

        if aux_columns is not None:
            if has_intercept and len(aux_columns) == n_gamma_cols - 1:
                gamma_col_names = ['intercept'] + list(aux_columns)
            elif len(aux_columns) == n_gamma_cols:
                gamma_col_names = list(aux_columns)
            else:
                gamma_col_names = (['intercept'] if has_intercept else []) + \
                    [f'aux_{j}' for j in range(n_gamma_cols - (1 if has_intercept else 0))]
        else:
            gamma_col_names = (['intercept'] if has_intercept else []) + \
                [f'aux_{j}' for j in range(n_gamma_cols - (1 if has_intercept else 0))]

        if label_columns is not None and len(label_columns) == model.mu_gamma.shape[0]:
            gamma_row_names = label_columns
        else:
            gamma_row_names = [f'outcome_{k}' for k in range(model.mu_gamma.shape[0])]

        gamma_df = pd.DataFrame(
            np.array(model.mu_gamma),
            index=gamma_row_names,
            columns=gamma_col_names
        )
        gamma_df.index.name = 'label'
        gamma_path = output_dir / f'{prefix}_gamma_weights{ext}'
        gamma_df.to_csv(gamma_path, compression=compression)
        saved_files['gamma_weights'] = gamma_path
        print(f"Saved gamma weights to {gamma_path}")

        # Also save gamma variance (diagonal of Sigma_gamma) for reference
        if hasattr(model, 'Sigma_gamma') and model.Sigma_gamma.size > 0:
            Sigma_gamma = np.array(model.Sigma_gamma)
            # Extract diagonal variances: (kappa, p_aux)
            gamma_var = np.stack([np.diag(Sigma_gamma[k]) for k in range(Sigma_gamma.shape[0])])
            gamma_var_df = pd.DataFrame(
                gamma_var,
                index=gamma_row_names,
                columns=gamma_col_names
            )
            gamma_var_df.index.name = 'label'
            gamma_var_path = output_dir / f'{prefix}_gamma_variance{ext}'
            gamma_var_df.to_csv(gamma_var_path, compression=compression)
            saved_files['gamma_variance'] = gamma_var_path
            print(f"Saved gamma variance to {gamma_var_path}")

    # Save summary
    summary = {
        'hyperparameters': {
            'n_factors': model.K,
            'a': model.a,
            'c': model.c,
            'b_v': model.b_v,
            'sigma_gamma': model.sigma_gamma,
        },
        'data_shapes': {
            f'n_{feature_type}s': len(gene_list),
            'n_train': len(splits['train']),
            'n_val': len(splits['val']),
            'n_test': len(splits['test']),
        },
        'feature_type': feature_type,
        'mode': mode,  # unmasked, masked, pathway_init, or combined
        'program_names': program_names,  # Pathway names if using pathway modes
        'n_pathway_factors': getattr(model, 'n_pathway_factors', None),  # For combined mode
        'training': {
            'final_elbo': model.elbo_history_[-1][1] if hasattr(model, 'elbo_history_') and model.elbo_history_ else None,
            'n_iterations': model.elbo_history_[-1][0] if hasattr(model, 'elbo_history_') and model.elbo_history_ else None,
            'final_holl': model.holl_history_[-1][1] if hasattr(model, 'holl_history_') and model.holl_history_ else None,
        },
        'label_columns': label_columns,
        'aux_columns': aux_columns,
        'classification': {
            'optimal_threshold': optimal_threshold,
        }
    }

    # === ELBO history ===
    if hasattr(model, 'elbo_history_'):
        summary['elbo_history'] = model.elbo_history_

    # === Held-out LL history ===
    if hasattr(model, 'holl_history_'):
        summary['holl_history'] = model.holl_history_

    # Convert JAX/NumPy types to JSON-serializable Python types
    def convert_to_json_serializable(obj):
        """Recursively convert JAX/NumPy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif hasattr(obj, '__array__'):  # JAX arrays and scalars
            arr = np.array(obj)
            if arr.ndim == 0:  # Scalar
                return arr.item()
            return arr.tolist()
        elif obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)  # Fallback for unknown types

    summary = convert_to_json_serializable(summary)

    summary_path = output_dir / f'{prefix}_summary.json'
    if compress:
        summary_path = output_dir / f'{prefix}_summary.json.gz'
        with gzip.open(summary_path, 'wt') as f:
            json.dump(summary, f, indent=2)
    else:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    saved_files['summary'] = summary_path
    print(f"Saved summary to {summary_path}")

    # Save training curves plot (ELBO + HO-LL with breakdowns)
    if (hasattr(model, 'elbo_history_') and model.elbo_history_) or \
       (hasattr(model, 'holl_history_') and model.holl_history_):
        plot_training_curves(model, save_dir=output_dir)
        saved_files['training_curves'] = output_dir / 'training_curves.png'

    return saved_files
