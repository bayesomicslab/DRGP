"""Emit one canonical AnnData plus a truth sidecar, for both modalities."""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


def to_anndata(counts, labels, background, truth, spec, aux=None):
    """counts: (n_units, n_genes); labels: dict from liability_label."""
    counts = np.asarray(counts)
    n_units = counts.shape[0]
    group = np.asarray(background.unit_to_group)
    obs = pd.DataFrame({
        "unit_id": [f"U{i}" for i in range(n_units)],
        "group_id": [f"G{g}" for g in group],
        "y": np.asarray(labels["y_unit"]).astype(int),
        "liability": np.asarray(labels["z_std"])[group] if spec.label.aggregate == "cell_mean"
                     else np.asarray(labels["z_std"]),
    }, index=[f"U{i}" for i in range(n_units)])
    if background.cell_type is not None:
        obs["cell_type"] = np.asarray(background.cell_type)
    if aux is not None:
        obs["aux"] = np.asarray(aux)
    var = pd.DataFrame(index=np.asarray(background.gene_ids).astype(str))
    A = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    A.uns["modality"] = spec.modality
    A.uns["polarity"] = np.array(truth.polarity)
    A.uns["program_type"] = np.array(truth.program_type)
    A.uns["upsilon"] = truth.upsilon
    A.uns["tau"] = float(labels["tau"])
    return A


def save_truth(truth, labels, path):
    """Sidecar with everything an evaluator needs. Object arrays are avoided so the file
    loads without allow_pickle."""
    members_flat = np.concatenate(truth.members)
    members_ptr = np.cumsum([0] + [len(m) for m in truth.members])
    np.savez_compressed(
        path,
        members_flat=members_flat, members_ptr=members_ptr,
        loadings=truth.loadings, upsilon=truth.upsilon,
        polarity=np.array(truth.polarity), program_type=np.array(truth.program_type),
        annotated_idx=truth.annotated_idx, mask=truth.mask,
        y=labels["y"], tau=labels["tau"])


def load_truth_members(npz):
    """Rebuild the per-program member lists from the flat/ptr encoding."""
    flat, ptr = npz["members_flat"], npz["members_ptr"]
    return [flat[ptr[i]:ptr[i + 1]] for i in range(len(ptr) - 1)]
