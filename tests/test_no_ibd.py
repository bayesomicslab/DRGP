"""The release must contain no trace of the dropped IBD/EMTAB cohort."""
import pathlib

FORBIDDEN = ("ibd", "emtab", "EMTAB11349")


def test_no_forbidden_cohort_strings():
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for f in root.rglob("*"):
        if not f.is_file() or ".git" in f.parts:
            continue
        if f.resolve() == pathlib.Path(__file__).resolve():
            continue  # this guard's own source names the forbidden terms
        if f.suffix.lower() not in {".py", ".md", ".toml", ".txt", ".r", ".sh", ".yaml", ".yml"}:
            continue
        text = f.read_text(errors="ignore").lower()
        for term in FORBIDDEN:
            if term.lower() in text:
                offenders.append(f"{f.relative_to(root)}: {term}")
    assert not offenders, "forbidden cohort references found:\n" + "\n".join(offenders)


def test_no_results_or_protected_data_committed():
    root = pathlib.Path(__file__).resolve().parents[1]
    for pat in ("*.h5ad", "*.npz", "*.csv.gz", "*.tsv.gz"):
        found = [p for p in root.rglob(pat) if ".git" not in p.parts]
        assert not found, f"data artifacts must not be committed: {found}"
