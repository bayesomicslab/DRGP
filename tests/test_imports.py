"""The model core must import with nothing outside drgp/ on the path."""


def test_model_core_imports():
    from drgp.model.cavi import CAVI
    from drgp.model.pg_kernel import pg_update_v
    from drgp.model import backend
    assert hasattr(CAVI, "fit")
    assert hasattr(CAVI, "transform")
    assert callable(pg_update_v)
    assert hasattr(backend, "xp")


def test_model_core_has_no_source_tree_dependency():
    """cavi.py must not reach back into the original package."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "drgp" / "model"
    for f in root.glob("*.py"):
        assert "VariationalInference" not in f.read_text(), f"{f.name} still references the source package"
