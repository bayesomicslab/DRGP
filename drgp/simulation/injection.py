"""Perturbation operators: plant program signal into the background counts.

Two irreducible implementations, selected by InjectionSpec.operator:

  'nb_resample' (single-cell) -- needs a ParametricBackground. Adds delta * (A_dev @ loadings^T)
      on the NATURAL-log mean and resamples the whole matrix from the fitted NB marginals.

  'thin_diff' (bulk) -- needs a RealizedBackground. Shells out to R (seqgendiff::thin_diff),
      binomial thinning on a LOG2 link, followed by thin_lib() equalization so that sequencing
      depth carries no phenotype association. REQUIRES R with the seqgendiff package.

The operators are not interchangeable: they consume different background shapes and different
link bases. This is the one place the two modalities genuinely cannot share code.
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd


def inject(background, loadings, activity, spec, rng, theta_base=0.0):
    if spec.operator == "nb_resample":
        return _nb_resample(background, loadings, activity, spec, rng, theta_base)
    if spec.operator == "thin_diff":
        return _thin_diff(background, loadings, activity, spec, rng)
    raise ValueError(f"unknown injection operator {spec.operator!r}")


def _nb_resample(background, loadings, activity, spec, rng, theta_base=0.0):
    if background.kind != "parametric":
        raise TypeError("nb_resample requires a ParametricBackground (mu + dispersion)")
    mu = np.asarray(background.mu, dtype=np.float32)
    A_dev = (np.asarray(activity, dtype=np.float32) - theta_base)
    # Perturb only genes carried by at least one program, matching the source implementation.
    carried = np.flatnonzero(np.abs(loadings).sum(axis=1) > 0)
    log_lam = np.log(np.maximum(mu, 1e-6))
    log_lam[:, carried] += spec.delta * (A_dev @ loadings[carried].T)
    lam = np.exp(log_lam, dtype=np.float32)
    n_arr = np.broadcast_to(np.asarray(background.dispersion, dtype=np.float32)[None, :], lam.shape)
    p = n_arr / (n_arr + np.maximum(lam, 1e-6))
    return rng.negative_binomial(n=n_arr, p=p).astype(np.int32)


def _thin_diff(background, loadings, activity, spec, rng):
    if background.kind != "realized":
        raise TypeError("thin_diff requires a RealizedBackground (realized counts)")
    rscript = os.environ.get("DRGP_RSCRIPT", "Rscript")
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "inject_thindiff.R")

    with tempfile.TemporaryDirectory() as td:
        genes = np.asarray(background.gene_ids).astype(str)
        samples = np.array([f"S{i}" for i in range(background.n_units)])
        # R script expects genes x samples
        pd.DataFrame(background.counts.T, index=genes, columns=samples).to_csv(
            f"{td}/X0.tsv.gz", sep="\t", compression="gzip")
        cols = [f"prog{k}" for k in range(loadings.shape[1])]
        pd.DataFrame(loadings, index=genes, columns=cols).to_csv(
            f"{td}/beta.tsv.gz", sep="\t", compression="gzip")
        pd.DataFrame(np.asarray(activity), index=samples, columns=cols).to_csv(
            f"{td}/theta.tsv.gz", sep="\t", compression="gzip")

        cmd = [rscript, script, "--x0", f"{td}/X0.tsv.gz", "--beta", f"{td}/beta.tsv.gz",
               "--theta", f"{td}/theta.tsv.gz", "--out", f"{td}/Xt.tsv.gz",
               "--seed", str(int(rng.integers(0, 2**31 - 1)))]
        if spec.equalize_library:
            cmd += ["--lib-frac", str(spec.lib_frac)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                "thin_diff injection failed: could not launch R. Bulk injection requires R with "
                f"the 'seqgendiff' package on PATH (set DRGP_RSCRIPT to point at Rscript).\n{exc}"
            ) from exc
        if res.returncode != 0:
            raise RuntimeError(
                "thin_diff injection failed. Bulk injection requires R with the 'seqgendiff' "
                f"package (set DRGP_RSCRIPT to point at it).\n{res.stderr[-2000:]}")
        Xt = pd.read_csv(f"{td}/Xt.tsv.gz", sep="\t", index_col=0)
        return Xt.reindex(index=genes, columns=samples).values.T.astype(np.int32)
