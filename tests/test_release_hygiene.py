"""Release hygiene: no result artifacts or protected data may be committed.

Only the intentionally shipped example data is exempt (simulated; contains no cohort or
patient information). Build artifacts are skipped because they are gitignored, not committed --
scanning them makes this fail for anyone who has run `pip install -e .`.
"""
import pathlib

_BUILD_DIRS = {"build", "dist", "__pycache__", ".pytest_cache", ".venv", "venv"}
_ALLOWED_DATA = {"data/toy_sim.h5ad"}


def _is_build_artifact(p):
    return any(part.endswith(".egg-info") or part in _BUILD_DIRS for part in p.parts)


def test_no_results_or_protected_data_committed():
    root = pathlib.Path(__file__).resolve().parents[1]
    for pat in ("*.h5ad", "*.npz", "*.csv.gz", "*.tsv.gz"):
        found = [p for p in root.rglob(pat)
                 if ".git" not in p.parts
                 and not _is_build_artifact(p)
                 and p.relative_to(root).as_posix() not in _ALLOWED_DATA]
        assert not found, f"data artifacts must not be committed: {found}"
