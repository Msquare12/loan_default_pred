def test_imports():
    # quick import smoke test
    import src.models.train as T
    assert hasattr(T, "train")
